"""Evaluation and ablations.

``hushkey eval`` reports WER for a manifest and, with ``--compare``, runs a second system
on the same utterances and a paired bootstrap test of the WER difference — the same
"is the signal real?" spirit as Subvox's permutation test. Typical questions:

* does my personal adapter beat the base model?      --adapter personal.pt --compare-adapter none
* does the throat/kinematic stream help at all?        --compare-zero-kinematics
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch

from .config import Paths, pick_device
from .data import UtteranceDataset, collate, read_manifest
from .model import Decoder, HushNet, ctc_greedy, load_rnnlm
from .model.mwer import word_edit_distance


def _system(paths: Paths, adapter: Optional[str], tiny: bool, device: str) -> HushNet:
    ap = None if adapter in (None, "none") else Path(adapter)
    net = HushNet.from_pretrained(None if tiny else paths.vsr_model, None if tiny else paths.vsr_conf, adapter_path=ap, tiny=tiny)
    return net.to(device).eval()


@torch.no_grad()
def per_utterance_errors(net: HushNet, ds: UtteranceDataset, device: str, beam: int, lm=None, zero_kin: bool = False, limit: int = 0):
    dec = Decoder(net, beam_size=beam, lm=lm).to(device) if beam > 0 else None
    rows = []
    n = len(ds) if not limit else min(limit, len(ds))
    for i in range(n):
        b = collate([ds[i]])
        kin = torch.zeros_like(b["kin"]) if zero_kin else b["kin"]
        hs, _ = net.encode(b["video"].to(device), kin.to(device))
        hyp = dec.beam(hs[0])[0].text if dec else ctc_greedy(net, hs[0])
        ref = b["texts"][0]
        rows.append((word_edit_distance(hyp, ref), len(ref.split()), ref, hyp))
    return rows


def wer(rows) -> float:
    return sum(r[0] for r in rows) / max(1, sum(r[1] for r in rows))


def paired_bootstrap(a, b, iters: int = 2000, seed: int = 0) -> float:
    """P(system B is not better than A) under resampling of utterances."""
    rng = random.Random(seed)
    n = len(a)
    worse = 0
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        ea, eb = sum(a[i][0] for i in idx), sum(b[i][0] for i in idx)
        worse += eb >= ea
    return worse / iters


def run_eval(manifest: str, adapter: Optional[str], beam: int, use_lm: bool, tiny: bool, device: str, limit: int, compare_adapter: Optional[str], compare_zero_kin: bool, show: int = 5) -> dict:
    paths = Paths().ensure()
    device = pick_device(device)
    ds = UtteranceDataset(read_manifest(manifest))
    net = _system(paths, adapter, tiny, device)
    lm = load_rnnlm(paths.lm_model, paths.lm_conf, net.odim) if (use_lm and beam > 0 and not tiny) else None
    a = per_utterance_errors(net, ds, device, beam, lm, limit=limit)
    report = {"system": {"adapter": adapter, "wer": wer(a), "n": len(a)}, "examples": [{"ref": r[2], "hyp": r[3]} for r in a[:show]]}
    if compare_adapter is not None or compare_zero_kin:
        if compare_zero_kin:
            b = per_utterance_errors(net, ds, device, beam, lm, zero_kin=True, limit=limit)
            label = "same system, kinematic stream zeroed"
        else:
            net_b = _system(paths, compare_adapter, tiny, device)
            b = per_utterance_errors(net_b, ds, device, beam, lm, limit=limit)
            label = f"adapter={compare_adapter}"
        report["baseline"] = {"label": label, "wer": wer(b)}
        # p-value that the main system is NOT better than the baseline
        report["p_value_main_not_better"] = paired_bootstrap(b, a)
    return report
