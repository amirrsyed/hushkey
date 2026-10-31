"""Minimum-word-error-rate (MWER) fine-tuning: sequence-level RL on the decoder's N-best.

For each utterance we sample an N-best list with beam search (no gradient), re-score the
hypotheses with gradients (attention decoder log-likelihood mixed with CTC), renormalise
over the list, and minimise the *expected* number of word errors relative to the list
mean (a REINFORCE baseline):

    L_mwer = Σ_i  P̂(y_i | x) · (W(y_i, y*) − W̄)          (Prabhavalkar et al., 2018)

The reward is the negative word edit distance to the reference, so the model is trained
on the metric dictation users actually feel, rather than token cross-entropy. As is
standard, we interpolate with the CE/CTC loss (λ) to keep training stable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .._vendor.espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from .._vendor.espnet.nets.pytorch_backend.transformer.mask import target_mask
from .hushnet import HushNet


def word_edit_distance(hyp: str, ref: str) -> int:
    h, r = hyp.split(), ref.split()
    d = list(range(len(r) + 1))
    for i in range(1, len(h) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(r) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (h[i - 1] != r[j - 1]))
            prev = cur
    return d[len(r)]


def sequence_logprobs(net: HushNet, hs: torch.Tensor, hyps: list[list[int]], ctc_weight: float) -> torch.Tensor:
    """Log p(y|x) for N hypotheses of ONE utterance. hs: (T, adim) with grad."""
    N = len(hyps)
    dev = hs.device
    L = max(1, max(len(h) for h in hyps))
    ys = torch.full((N, L), -1, dtype=torch.long, device=dev)
    for i, h in enumerate(hyps):
        if h:
            ys[i, : len(h)] = torch.tensor(h, device=dev)
    mem = hs.unsqueeze(0).expand(N, -1, -1)
    ys_in, ys_out = add_sos_eos(ys, net.sos, net.eos, -1)
    pred, _ = net.backbone.decoder(ys_in, target_mask(ys_in, -1), mem, None)
    logp = F.log_softmax(pred, dim=-1)
    valid = ys_out != -1
    tok = logp.gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    att = (tok * valid).sum(-1)
    if ctc_weight <= 0:
        return att
    lp = F.log_softmax(net.backbone.ctc.ctc_lo(hs), dim=-1)  # (T, V)
    T = hs.size(0)
    ctc = []
    for h in hyps:
        if not h:
            ctc.append(lp[:, 0].sum())
            continue
        ctc.append(
            -F.ctc_loss(lp.unsqueeze(1), torch.tensor([h], device=dev), torch.tensor([T]), torch.tensor([len(h)]), blank=0, reduction="sum", zero_infinity=True)
        )
    return (1 - ctc_weight) * att + ctc_weight * torch.stack(ctc)


def mwer_loss(net: HushNet, hs: torch.Tensor, nbest_tokens: list[list[int]], nbest_texts: list[str], ref_text: str, ctc_weight: float = 0.1):
    errs = torch.tensor([float(word_edit_distance(t, ref_text)) for t in nbest_texts], device=hs.device)
    logp = sequence_logprobs(net, hs, nbest_tokens, ctc_weight)
    post = torch.softmax(logp, dim=0)
    loss = (post * (errs - errs.mean())).sum()
    return loss, {"expected_errors": float((post.detach() * errs).sum()), "oracle_errors": float(errs.min()), "top1_errors": float(errs[0])}
