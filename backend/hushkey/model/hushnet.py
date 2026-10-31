"""HushNet: Auto-AVSR VSR + LoRA adapters + gated kinematic (lip/jaw/throat) fusion.

    mouth video ─► Conv3d+ResNet18 frontend ─► 512-d per frame ─┐
                                                                 (+) ─► linear+relpos ─► 12× conformer ─► CTC / decoder
    kinematics (16-d) ─► KinematicEncoder ─► 512-d ─► × tanh(g) ┘

``g`` is a learned per-channel gate initialised to 0, so an untrained HushNet is exactly
the pretrained Auto-AVSR model; adaptation can only *add* information from the
kinematic stream if it lowers the loss. The backbone stays frozen; only LoRA weights,
the kinematic encoder and the gate are trained (≈1–2 % of parameters), so a personal
adapter is a few MB and trains on a laptop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .._vendor.espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from .._vendor.espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from .._vendor.espnet.nets.pytorch_backend.transformer.mask import target_mask
from ..preprocess.keypoints import KIN_DIM
from .backbone import load_backbone
from .lora import DEFAULT_TARGETS, LoRALinear, inject_lora


class KinematicEncoder(nn.Module):
    """Small temporal CNN over the kinematic stream (no temporal subsampling)."""

    def __init__(self, in_dim: int = KIN_DIM, hidden: int = 128, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, out_dim, 1),
        )

    def forward(self, kin: torch.Tensor) -> torch.Tensor:  # (B, T, D) → (B, T, out)
        return self.net(kin.transpose(1, 2)).transpose(1, 2)


class GatedFusion(nn.Module):
    def __init__(self, dim: int = 512, in_dim: int = KIN_DIM):
        super().__init__()
        self.enc = KinematicEncoder(in_dim, out_dim=dim)
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, vis: torch.Tensor, kin: Optional[torch.Tensor]) -> torch.Tensor:
        if kin is None:
            return vis
        T = min(vis.size(1), kin.size(1))
        add = self.enc(kin[:, :T]) * torch.tanh(self.gate)
        return torch.cat([vis[:, :T] + add, vis[:, T:]], dim=1)


class HushNet(nn.Module):
    def __init__(self, backbone, token_list, args, lora_rank: int = 8, lora_alpha: float = 16.0, use_kinematics: bool = True, lora_targets=DEFAULT_TARGETS):
        super().__init__()
        self.backbone = backbone
        self.token_list = token_list
        self.args = args
        self.ctc_weight = float(getattr(args, "mtlalpha", 0.1))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.lora_targets = inject_lora(self.backbone, lora_targets, r=lora_rank, alpha=lora_alpha) if lora_rank > 0 else []
        self.fusion = GatedFusion(512) if use_kinematics else None
        self.meta = {"lora_rank": lora_rank, "lora_alpha": lora_alpha, "use_kinematics": use_kinematics}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(cls, model_path=None, conf_path=None, adapter_path=None, tiny=False, **kw) -> "HushNet":
        backbone, token_list, args, loaded = load_backbone(model_path, conf_path, tiny=tiny)
        if adapter_path is not None and Path(adapter_path).exists():
            meta = torch.load(adapter_path, map_location="cpu", weights_only=True).get("meta", {})
            kw = {**{k: meta[k] for k in ("lora_rank", "lora_alpha", "use_kinematics") if k in meta}, **kw}
        net = cls(backbone, token_list, args, **kw)
        net.backbone_loaded = loaded
        if adapter_path is not None and Path(adapter_path).exists():
            net.load_adapters(adapter_path)
        return net

    @property
    def odim(self) -> int:
        return self.backbone.odim

    @property
    def sos(self) -> int:
        return self.backbone.sos

    @property
    def eos(self) -> int:
        return self.backbone.eos

    # ------------------------------------------------------------------ adapters
    def adapter_state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if ".lora_" in k or k.startswith("fusion.")}

    def save_adapters(self, path, extra: Optional[dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.adapter_state_dict(), "meta": {**self.meta, **(extra or {})}}, path)

    def load_adapters(self, path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(blob["state"], strict=False)
        bad = [k for k in unexpected]
        if bad:
            raise RuntimeError(f"adapter file has unexpected keys (architecture mismatch?): {bad[:5]}")

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ forward
    def encode(self, video: torch.Tensor, kin: Optional[torch.Tensor], lengths: Optional[torch.Tensor] = None):
        """video: (B, T, 88, 88) normalised; kin: (B, T, D) or None; lengths: (B,).

        Returns (hs (B, T, adim), hs_mask (B, 1, T) or None).
        """
        enc = self.backbone.encoder
        x = video.unsqueeze(1)  # (B, 1, T, H, W)
        feats = enc.frontend(x)  # (B, T, 512)
        if self.fusion is not None:
            feats = self.fusion(feats, kin)
        mask = None
        if lengths is not None:
            mask = make_non_pad_mask(lengths.cpu()).to(video.device).unsqueeze(-2)
        xs = enc.embed(feats)
        xs, mask = enc.encoders(xs, mask)
        if isinstance(xs, tuple):
            xs = xs[0]
        if enc.normalize_before:
            xs = enc.after_norm(xs)
        return xs, mask

    def loss(self, video, kin, lengths, ys_pad):
        """Joint CTC/attention loss used by Auto-AVSR (ctc_weight = mtlalpha, default 0.1)."""
        hs, hs_mask = self.encode(video, kin, lengths)
        hlens = lengths.to(hs.device)
        loss_ctc = self.backbone.ctc(hs, hlens, ys_pad)
        ys_in, ys_out = add_sos_eos(ys_pad, self.sos, self.eos, -1)
        ys_mask = target_mask(ys_in, -1)
        pred, _ = self.backbone.decoder(ys_in, ys_mask, hs, hs_mask)
        loss_att = self.backbone.criterion(pred, ys_out)
        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        valid = ys_out != -1
        acc = (pred.argmax(-1)[valid] == ys_out[valid]).float().mean()
        return {"loss": loss, "ctc": loss_ctc.detach(), "att": loss_att.detach(), "acc": acc.detach()}

    def lora_layers(self):
        return [m for m in self.modules() if isinstance(m, LoRALinear)]
