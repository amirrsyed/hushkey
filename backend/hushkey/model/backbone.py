"""Load the pretrained Auto-AVSR visual speech recognition model.

Default checkpoint: ``LRS3_V_WER19.1`` — the Auto-AVSR VSR conformer (≈250 M params,
19.1 % WER on LRS3 test) trained on LRS3 + VoxCeleb2 + AVSpeech with automatic labels,
the same checkpoint Chaplin uses. ``hushkey download`` fetches it from Chaplin's
Hugging Face mirror.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from .._vendor.espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from ..config import TOKENS_FILE

#: Architecture of LRS3_V_WER19.1 (used when model.json is absent, e.g. in tests).
AUTO_AVSR_VSR_CONF = dict(
    adim=768,
    aheads=12,
    eunits=3072,
    elayers=12,
    dlayers=6,
    dunits=3072,
    transformer_input_layer="conv3d",
    transformer_encoder_attn_layer_type="rel_mha",
    rel_pos_type="latest",
    macaron_style=True,
    use_cnn_module=True,
    cnn_module_kernel=31,
    zero_triu=False,
    a_upsample_ratio=1,
    relu_type="swish",
    dropout_rate=0.1,
    transformer_attn_dropout_rate=0.1,
    transformer_length_normalized_loss=False,
    mtlalpha=0.1,
    lsm_weight=0.1,
    ctc_type="builtin",
    report_cer=False,
    report_wer=False,
    labels_type="unigram5000",
)

#: A ~1 M-parameter model with the same structure, for CI and smoke tests.
TINY_CONF = dict(AUTO_AVSR_VSR_CONF, adim=64, aheads=4, eunits=128, elayers=2, dlayers=1, dunits=128, cnn_module_kernel=7)


def load_token_list() -> list[str]:
    units = TOKENS_FILE.read_text(encoding="utf8").splitlines()
    return ["<blank>"] + [u.split()[0] for u in units] + ["<eos>"]


def read_conf(conf_path: Optional[Path]) -> dict:
    if conf_path is None or not Path(conf_path).exists():
        return dict(AUTO_AVSR_VSR_CONF)
    confs = json.loads(Path(conf_path).read_text())
    args = confs if isinstance(confs, dict) else confs[2]
    merged = dict(AUTO_AVSR_VSR_CONF)
    merged.update(args)
    return merged


def build_backbone(conf: dict, token_list: Optional[list[str]] = None) -> tuple[E2E, list[str], argparse.Namespace]:
    token_list = token_list or load_token_list()
    args = argparse.Namespace(**conf)
    args.char_list = token_list
    model = E2E(len(token_list), args)
    return model, token_list, args


def load_backbone(model_path: Optional[Path], conf_path: Optional[Path] = None, tiny: bool = False):
    """Build the E2E model and load weights if ``model_path`` exists (random init otherwise)."""
    conf = dict(TINY_CONF) if tiny else read_conf(conf_path)
    if tiny:
        # The tiny model has no checkpoint: make its random weights identical in every process,
        # so adapters trained against it (tests, UI dev) still match when the server reloads it.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            model, token_list, args = build_backbone(conf)
        return model, token_list, args, False
    model, token_list, args = build_backbone(conf)
    loaded = False
    if model_path is not None and Path(model_path).exists():
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        loaded = True
    return model, token_list, args, loaded
