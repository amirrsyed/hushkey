"""Minimal LoRA (Hu et al., 2021) for nn.Linear, no external dependencies."""

from __future__ import annotations

import math
import re
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))  # zero → starts as identity
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    @property
    def weight(self):  # some espnet code reads .weight
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.enabled:
            y = y + (self.dropout(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scale
        return y

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        lin.weight.copy_(self.base.weight + (self.lora_b @ self.lora_a) * self.scale)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin.to(self.base.weight.device)


#: Encoder attention + feed-forward projections, and decoder self/src attention.
DEFAULT_TARGETS = (
    r"encoder\.encoders\.\d+\.self_attn\.linear_(q|k|v|out)$",
    r"encoder\.encoders\.\d+\.feed_forward(_macaron)?\.w_[12]$",
    r"decoder\.decoders\.\d+\.(self_attn|src_attn)\.linear_(q|v)$",
)


def inject_lora(model: nn.Module, patterns: Iterable[str] = DEFAULT_TARGETS, r: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> list[str]:
    regs = [re.compile(p) for p in patterns]
    targets = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and any(rg.search(n) for rg in regs)]
    for name in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(getattr(parent, child), r=r, alpha=alpha, dropout=dropout))
    return targets


def lora_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for m in lora_modules(model):
        m.enabled = enabled
