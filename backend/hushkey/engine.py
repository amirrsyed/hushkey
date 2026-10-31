"""Inference runtime shared by the server and the CLI."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .cleanup import CleanupConfig, finalize
from .config import Paths, pick_device
from .model import Decoder, HushNet, ctc_greedy, load_rnnlm
from .preprocess import Sample
from .preprocess.mouth import to_model_input

log = logging.getLogger("hushkey.engine")

ADAPTER_ORDER = ("personal.pt", "silent.pt")


@dataclass
class EngineConfig:
    device: str = "auto"
    beam_size: int = 10
    ctc_weight: float = 0.1
    lm_weight: float = 0.3
    use_lm: bool = True
    adapter: Optional[str] = None  # path, or None = auto-pick from ~/…/Hushkey/adapters
    tiny: bool = False  # random tiny model (tests / UI development without weights)
    min_frames: int = 12  # ~0.5 s at 25 fps
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)


class Engine:
    def __init__(self, cfg: EngineConfig, paths: Optional[Paths] = None):
        self.cfg = cfg
        self.paths = (paths or Paths()).ensure()
        self.device = pick_device(cfg.device)
        self.lock = threading.Lock()
        adapter = Path(cfg.adapter) if cfg.adapter else next((self.paths.adapters / n for n in ADAPTER_ORDER if (self.paths.adapters / n).exists()), None)
        self.adapter_path = adapter
        t0 = time.time()
        self.net = HushNet.from_pretrained(
            None if cfg.tiny else self.paths.vsr_model,
            None if cfg.tiny else self.paths.vsr_conf,
            adapter_path=adapter,
            tiny=cfg.tiny,
        )
        if not cfg.tiny and not self.net.backbone_loaded:
            raise FileNotFoundError(f"VSR weights not found at {self.paths.vsr_model}. Run `hushkey download` first (or pass --tiny for a random test model).")
        self.net.to(self.device).eval()
        lm = load_rnnlm(self.paths.lm_model, self.paths.lm_conf, self.net.odim) if (cfg.use_lm and not cfg.tiny) else None
        self.has_lm = lm is not None
        self.decoder = Decoder(self.net, beam_size=cfg.beam_size, ctc_weight=cfg.ctc_weight, lm=lm, lm_weight=cfg.lm_weight).to(self.device)
        log.info("engine ready in %.1fs on %s (adapter=%s, lm=%s)", time.time() - t0, self.device, adapter, self.has_lm)

    def info(self) -> dict:
        return {
            "device": self.device,
            "adapter": str(self.adapter_path) if self.adapter_path else None,
            "lm": self.has_lm,
            "tiny": self.cfg.tiny,
            "beam_size": self.cfg.beam_size,
            "kinematics": self.net.fusion is not None,
            "kinematic_gate": float(torch.tanh(self.net.fusion.gate.detach()).abs().mean()) if self.net.fusion is not None else 0.0,
        }

    def _tensors(self, sample: Sample):
        x = torch.from_numpy(to_model_input(sample.rois)).unsqueeze(0).to(self.device)
        k = torch.from_numpy(sample.kin).unsqueeze(0).to(self.device)
        return x, k

    @torch.no_grad()
    def partial(self, sample: Sample) -> str:
        if len(sample.rois) < self.cfg.min_frames:
            return ""
        with self.lock:
            x, k = self._tensors(sample)
            hs, _ = self.net.encode(x, k)
            return ctc_greedy(self.net, hs[0])

    @torch.no_grad()
    def transcribe(self, sample: Sample, nbest: int = 4, app: Optional[str] = None) -> dict:
        t0 = time.time()
        if len(sample.rois) < self.cfg.min_frames:
            return {"raw": "", "text": "", "alternatives": [], "llm": False, "frames": len(sample.rois), "latency_ms": 0, "reason": "too_short"}
        with self.lock:
            x, k = self._tensors(sample)
            hs, _ = self.net.encode(x, k)
            hyps = self.decoder.beam(hs[0], nbest=nbest)
        t_vsr = time.time()
        raw = hyps[0].text if hyps else ""
        alts = [h.text for h in hyps[1:]]
        text, used_llm = finalize(raw, alts, self.cfg.cleanup, app) if raw else ("", False)
        return {
            "raw": raw,
            "text": text,
            "alternatives": alts,
            "llm": used_llm,
            "frames": int(len(sample.rois)),
            "face_ratio": round(sample.face_ratio, 3),
            "vsr_ms": int((t_vsr - t0) * 1000),
            "latency_ms": int((time.time() - t0) * 1000),
        }
