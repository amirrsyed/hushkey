"""Filesystem layout and runtime defaults."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_DIR / "assets"
TOKENS_FILE = ASSETS_DIR / "unigram5000_units.txt"
SPM_MODEL_FILE = ASSETS_DIR / "unigram5000.model"
MEAN_FACE_FILE = ASSETS_DIR / "20words_mean_face.npy"

#: The Auto-AVSR models were trained on 25 fps video.
MODEL_FPS = 25.0


def hushkey_home() -> Path:
    env = os.environ.get("HUSHKEY_HOME")
    if env:
        return Path(env).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Hushkey"
    return Path.home() / ".hushkey"


@dataclass
class Paths:
    home: Path = field(default_factory=hushkey_home)

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def vsr_model(self) -> Path:
        return self.models / "LRS3_V_WER19.1" / "model.pth"

    @property
    def vsr_conf(self) -> Path:
        return self.models / "LRS3_V_WER19.1" / "model.json"

    @property
    def lm_model(self) -> Path:
        return self.models / "lm_en_subword" / "model.pth"

    @property
    def lm_conf(self) -> Path:
        return self.models / "lm_en_subword" / "model.json"

    @property
    def adapters(self) -> Path:
        return self.home / "adapters"

    @property
    def enrollment(self) -> Path:
        return self.home / "enrollment"

    def ensure(self) -> "Paths":
        for p in (self.models, self.adapters, self.enrollment):
            p.mkdir(parents=True, exist_ok=True)
        return self


def pick_device(requested: str = "auto") -> str:
    """Pick a torch device string.

    ``auto`` prefers CUDA, then CPU. Apple's MPS backend is opt-in (``--device mps``)
    because a few ops in the CTC prefix scorer fall back to CPU and can be slower
    than plain CPU for short clips.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
