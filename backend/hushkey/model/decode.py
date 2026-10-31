"""Decoding: joint CTC/attention beam search (+ optional RNN LM), and fast CTC-greedy partials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .._vendor.espnet.asr.asr_utils import get_model_conf, torch_load
from .._vendor.espnet.nets.batch_beam_search import BatchBeamSearch
from .._vendor.espnet.nets.lm_interface import dynamic_import_lm
from .._vendor.espnet.nets.scorers.length_bonus import LengthBonus
from .hushnet import HushNet


@dataclass
class Hyp:
    tokens: list[int]  # without sos/eos
    text: str
    score: float


def tokens_to_text(tokens, token_list) -> str:
    pieces = [token_list[t] for t in tokens if 0 < t < len(token_list) - 1]
    return "".join(pieces).replace("▁", " ").replace("<unk>", "").strip()


def load_rnnlm(lm_path: Optional[Path], lm_conf: Optional[Path], vocab: int):
    if lm_path is None or not Path(lm_path).exists():
        return None
    lm_args = get_model_conf(str(lm_path), str(lm_conf) if lm_conf else None)
    lm_class = dynamic_import_lm(getattr(lm_args, "model_module", "default"), getattr(lm_args, "backend", "pytorch"))
    lm = lm_class(vocab, lm_args)
    torch_load(str(lm_path), lm)
    return lm.eval()


class Decoder:
    def __init__(self, net: HushNet, beam_size: int = 10, ctc_weight: float = 0.1, lm=None, lm_weight: float = 0.3, penalty: float = 0.0):
        self.net = net
        scorers = net.backbone.scorers()
        scorers["lm"] = lm
        scorers["length_bonus"] = LengthBonus(net.odim)
        weights = dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, lm=lm_weight if lm is not None else 0.0, length_bonus=penalty)
        self.search = BatchBeamSearch(
            beam_size=beam_size,
            vocab_size=net.odim,
            weights=weights,
            scorers=scorers,
            sos=net.sos,
            eos=net.eos,
            token_list=net.token_list,
            pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
        )
        self.search.eval()

    def to(self, device):
        self.search.to(device)
        return self

    @torch.no_grad()
    def beam(self, hs: torch.Tensor, nbest: int = 1) -> list[Hyp]:
        """hs: (T, adim) encoder output for a single utterance."""
        hyps = self.search(hs)
        out = []
        for h in hyps[:nbest]:
            toks = [int(t) for t in h.yseq[1:]]
            if toks and toks[-1] == self.net.eos:
                toks = toks[:-1]
            out.append(Hyp(tokens=toks, text=tokens_to_text(toks, self.net.token_list), score=float(h.score)))
        return out


@torch.no_grad()
def ctc_greedy(net: HushNet, hs: torch.Tensor) -> str:
    """Collapse-repeats/remove-blanks argmax over CTC posteriors. hs: (T, adim)."""
    ids = net.backbone.ctc.argmax(hs.unsqueeze(0))[0].tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != 0:
            out.append(i)
        prev = i
    return tokens_to_text(out, net.token_list)
