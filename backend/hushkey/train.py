"""Three training stages, all of which train adapters only (the 250 M backbone stays frozen).

stage ``adapt``        public talking-face corpora (LRS3 / VoxCeleb2 / your own) with
                       silent-speech augmentation → ``adapters/silent.pt``
stage ``personalize``  your enrollment clips from the app (+ optional replay of public
                       data so nothing is forgotten) → ``adapters/personal.pt``
stage ``mwer``         sequence-level RL (minimum word error rate) on N-best lists
                       from beam search, starting from the adapter above
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .config import Paths, pick_device
from .data import FrameBudgetSampler, UtteranceDataset, collate, read_manifest, split_rows
from .model import Decoder, HushNet, ctc_greedy
from .model.mwer import mwer_loss, word_edit_distance
from .preprocess.augment import SilentSpeechAugment

log = logging.getLogger("hushkey.train")


@dataclass
class TrainConfig:
    stage: str = "adapt"
    manifests: list[str] = field(default_factory=list)
    replay_manifests: list[str] = field(default_factory=list)
    replay_ratio: float = 0.3
    out: Optional[str] = None
    init_adapter: Optional[str] = None
    steps: int = 20000
    lr: float = 5e-4
    warmup: int = 500
    weight_decay: float = 0.01
    max_frames: int = 1600
    grad_clip: float = 5.0
    lora_rank: int = 8
    use_kinematics: bool = True
    val_fraction: float = 0.05
    eval_every: int = 500
    eval_max: int = 200
    device: str = "auto"
    seed: int = 0
    tiny: bool = False
    workers: int = 2
    log_every: int = 25
    # mwer
    nbest: int = 4
    mwer_ce_weight: float = 0.1
    mwer_beam: int = 4

    @classmethod
    def for_stage(cls, stage: str, **kw) -> "TrainConfig":
        presets = {
            "adapt": dict(steps=20000, lr=5e-4, warmup=500),
            "personalize": dict(steps=600, lr=3e-4, warmup=50, val_fraction=0.15, eval_every=100, max_frames=800),
            "mwer": dict(steps=300, lr=5e-5, warmup=20, val_fraction=0.15, eval_every=50),
        }
        base = presets.get(stage, {})
        base.update({k: v for k, v in kw.items() if v is not None})
        return cls(stage=stage, **base)


def _augment_for(stage: str, seed: int) -> SilentSpeechAugment:
    if stage == "adapt":
        return SilentSpeechAugment(seed=seed)
    # enrollment clips are already silent webcam speech: only mild nuisance augmentation
    return SilentSpeechAugment(p_attenuate=0.15, alpha=(0.8, 1.0), p_tempo=0.3, tempo_range=(0.9, 1.1), p_webcam=0.5, p_kin_drop=0.2, seed=seed)


def _cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def build_net(cfg: TrainConfig, paths: Paths) -> HushNet:
    init = Path(cfg.init_adapter) if cfg.init_adapter else None
    if init is None and cfg.stage in ("personalize", "mwer"):
        for name in (("personal.pt", "silent.pt") if cfg.stage == "mwer" else ("silent.pt",)):
            if (paths.adapters / name).exists():
                init = paths.adapters / name
                break
    kw = {} if init else {"lora_rank": cfg.lora_rank, "use_kinematics": cfg.use_kinematics}
    net = HushNet.from_pretrained(None if cfg.tiny else paths.vsr_model, None if cfg.tiny else paths.vsr_conf, adapter_path=init, tiny=cfg.tiny, **kw)
    if not cfg.tiny and not net.backbone_loaded:
        raise FileNotFoundError("Pretrained VSR weights missing: run `hushkey download` first.")
    log.info("initial adapter: %s", init)
    return net


@torch.no_grad()
def evaluate_wer(net: HushNet, ds: UtteranceDataset, device: str, limit: int = 200, beam: int = 0, zero_kin: bool = False) -> dict:
    """WER on up to `limit` utterances. beam=0 → fast CTC-greedy; beam>0 → joint beam search (no LM)."""
    net.eval()
    dec = Decoder(net, beam_size=beam).to(device) if beam > 0 else None
    errs = words = 0
    examples = []
    for i in range(min(limit, len(ds))):
        b = collate([ds[i]])
        kin = torch.zeros_like(b["kin"]) if zero_kin else b["kin"]
        hs, _ = net.encode(b["video"].to(device), kin.to(device))
        hyp = dec.beam(hs[0])[0].text if dec else ctc_greedy(net, hs[0])
        ref = b["texts"][0]
        errs += word_edit_distance(hyp, ref)
        words += max(1, len(ref.split()))
        if len(examples) < 5:
            examples.append({"ref": ref, "hyp": hyp})
    return {"wer": errs / max(1, words), "n": min(limit, len(ds)), "examples": examples}


def _loaders(cfg: TrainConfig, rows_train, rows_replay):
    aug = _augment_for(cfg.stage, cfg.seed)
    ds = UtteranceDataset(rows_train, augment=aug)
    dl = DataLoader(ds, batch_sampler=FrameBudgetSampler(ds, cfg.max_frames, seed=cfg.seed), collate_fn=collate, num_workers=cfg.workers)
    dl_replay = None
    if rows_replay:
        ds_r = UtteranceDataset(rows_replay, augment=_augment_for("adapt", cfg.seed + 1))
        dl_replay = DataLoader(ds_r, batch_sampler=FrameBudgetSampler(ds_r, cfg.max_frames, seed=cfg.seed + 1), collate_fn=collate, num_workers=cfg.workers)
    return ds, dl, dl_replay


def _forever(dl):
    while True:
        for b in dl:
            yield b


def train(cfg: TrainConfig, paths: Optional[Paths] = None) -> dict:
    paths = (paths or Paths()).ensure()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    out = Path(cfg.out) if cfg.out else paths.adapters / {"adapt": "silent.pt", "personalize": "personal.pt", "mwer": "personal.pt"}[cfg.stage]

    rows = [r for m in cfg.manifests for r in read_manifest(m)]
    if not rows:
        raise SystemExit("no training utterances found in the given manifests")
    rows_train, rows_val = split_rows(rows, cfg.val_fraction, cfg.seed)
    rows_replay = [r for m in cfg.replay_manifests for r in read_manifest(m)]
    log.info("train=%d val=%d replay=%d device=%s", len(rows_train), len(rows_val), len(rows_replay), device)

    net = build_net(cfg, paths).to(device)
    params = net.trainable_parameters()
    log.info("trainable params: %.2fM of %.1fM", sum(p.numel() for p in params) / 1e6, sum(p.numel() for p in net.parameters()) / 1e6)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _cosine(s, cfg.warmup, cfg.steps))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    val_ds = UtteranceDataset(rows_val) if rows_val else None
    history = {"config": asdict(cfg), "evals": []}
    if val_ds:
        base = evaluate_wer(net, val_ds, device, cfg.eval_max, beam=cfg.mwer_beam if cfg.stage == "mwer" else 0)
        history["evals"].append({"step": 0, **base})
        log.info("step 0 val WER %.3f", base["wer"])
    best = history["evals"][0]["wer"] if history["evals"] else float("inf")

    if cfg.stage == "mwer":
        result = _train_mwer(cfg, net, rows_train, opt, sched, device, val_ds, history, out, best)
        return result

    ds, dl, dl_replay = _loaders(cfg, rows_train, rows_replay)
    it, it_r = _forever(dl), (_forever(dl_replay) if dl_replay else None)
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        net.train()
        b = next(it_r) if (it_r is not None and random.random() < cfg.replay_ratio) else next(it)
        with torch.autocast("cuda", enabled=use_amp):
            o = net.loss(b["video"].to(device), b["kin"].to(device), b["lengths"].to(device), b["targets"].to(device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(o["loss"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % cfg.log_every == 0:
            log.info("step %d loss %.3f ctc %.2f att %.2f acc %.3f lr %.2e %.1fs", step, o["loss"].item(), o["ctc"], o["att"], o["acc"], sched.get_last_lr()[0], time.time() - t0)
        if val_ds and (step % cfg.eval_every == 0 or step == cfg.steps):
            ev = evaluate_wer(net, val_ds, device, cfg.eval_max)
            history["evals"].append({"step": step, **ev})
            log.info("step %d val WER %.3f (best %.3f)", step, ev["wer"], best)
            if ev["wer"] <= best:
                best = ev["wer"]
                net.save_adapters(out, {"stage": cfg.stage, "step": step, "val_wer": best})
    if not val_ds:
        net.save_adapters(out, {"stage": cfg.stage, "step": cfg.steps})
    history["best_wer"] = best
    history["adapter"] = str(out)
    out.with_suffix(".json").write_text(json.dumps(history, indent=2))
    return history


def _train_mwer(cfg, net, rows_train, opt, sched, device, val_ds, history, out, best):
    ds = UtteranceDataset(rows_train, augment=_augment_for("personalize", cfg.seed))
    dec = Decoder(net, beam_size=cfg.mwer_beam).to(device)
    order = list(range(len(ds)))
    params = net.trainable_parameters()
    for step in range(1, cfg.steps + 1):
        if not order:
            order = list(range(len(ds)))
        random.shuffle(order)
        b = collate([ds[order.pop()]])
        video, kin, lengths, ys = (b[k].to(device) for k in ("video", "kin", "lengths", "targets"))
        # 1) N-best from the current policy (no grad, eval mode for deterministic dropout)
        net.eval()
        with torch.no_grad():
            hs0, _ = net.encode(video, kin)
            hyps = dec.beam(hs0[0], nbest=cfg.nbest)
        if len(hyps) < 2:
            continue
        # 2) re-score with grad, expected-WER objective + CE anchor
        net.train()
        hs, _ = net.encode(video, kin)
        l_mwer, st = mwer_loss(net, hs[0], [h.tokens for h in hyps], [h.text for h in hyps], b["texts"][0], net.ctc_weight)
        l_ce = net.loss(video, kin, lengths, ys)["loss"] if cfg.mwer_ce_weight > 0 else torch.zeros((), device=device)
        loss = l_mwer + cfg.mwer_ce_weight * l_ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        if step % 10 == 0:
            log.info("mwer step %d loss %.3f E[err] %.2f top1 %.0f oracle %.0f", step, loss.item(), st["expected_errors"], st["top1_errors"], st["oracle_errors"])
        if val_ds and (step % cfg.eval_every == 0 or step == cfg.steps):
            ev = evaluate_wer(net, val_ds, device, cfg.eval_max, beam=cfg.mwer_beam)
            history["evals"].append({"step": step, **ev})
            log.info("mwer step %d val WER(beam) %.3f", step, ev["wer"])
            if ev["wer"] <= best:
                best = ev["wer"]
                net.save_adapters(out, {"stage": "mwer", "step": step, "val_wer": best})
    if not val_ds:
        net.save_adapters(out, {"stage": "mwer", "step": cfg.steps})
    history["best_wer"] = best
    history["adapter"] = str(out)
    out.with_suffix(".json").write_text(json.dumps(history, indent=2))
    return history
