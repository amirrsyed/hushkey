"""Datasets of preprocessed utterances (``.npz`` written by ``Sample.save``).

A manifest is JSON Lines; each line is ``{"path": "relative/or/absolute.npz", "text": "...", "frames": T}``.
LRS3, VoxCeleb2, your own recordings and the app's enrollment clips all share it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..model.tokenizer import Tokenizer, normalize_text
from ..preprocess import Sample
from ..preprocess.augment import SilentSpeechAugment
from ..preprocess.mouth import to_model_input


def read_manifest(path) -> list[dict]:
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p = Path(r["path"])
        r["path"] = str(p if p.is_absolute() else path.parent / p)
        rows.append(r)
    return rows


def split_rows(rows: list[dict], val_fraction: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_val = int(round(len(rows) * val_fraction)) if len(rows) > 4 else 0
    return rows[n_val:], rows[:n_val]


class UtteranceDataset(Dataset):
    def __init__(self, rows: list[dict], augment: Optional[SilentSpeechAugment] = None, tokenizer: Optional[Tokenizer] = None, max_frames: int = 600):
        self.rows = [r for r in rows if r.get("frames", 1) <= max_frames]
        self.augment = augment
        self.tok = tokenizer or Tokenizer()

    def __len__(self) -> int:
        return len(self.rows)

    def frames(self, i: int) -> int:
        return int(self.rows[i].get("frames", 75))

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        sample, text = Sample.load(r["path"])
        text = normalize_text(r.get("text") or text)
        if self.augment is not None:
            video, kin = self.augment(sample.rois, sample.kin)
        else:
            video, kin = to_model_input(sample.rois), sample.kin
        return {"video": torch.from_numpy(np.ascontiguousarray(video)), "kin": torch.from_numpy(np.ascontiguousarray(kin)), "target": self.tok.encode(text), "text": text}


def collate(batch: list[dict]) -> dict:
    B = len(batch)
    T = max(b["video"].shape[0] for b in batch)
    L = max(1, max(len(b["target"]) for b in batch))
    H, W = batch[0]["video"].shape[1:]
    D = batch[0]["kin"].shape[1]
    video = torch.zeros(B, T, H, W)
    kin = torch.zeros(B, T, D)
    ys = torch.full((B, L), -1, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["video"].shape[0]
        video[i, :t] = b["video"]
        kin[i, :t] = b["kin"][:t]
        ys[i, : len(b["target"])] = b["target"]
        lengths[i] = t
    return {"video": video, "kin": kin, "lengths": lengths, "targets": ys, "texts": [b["text"] for b in batch]}


class FrameBudgetSampler(Sampler[list[int]]):
    """Batches of similar-length utterances whose padded size stays under ``max_frames`` (like Auto-AVSR's bucketing)."""

    def __init__(self, ds: UtteranceDataset, max_frames: int = 1600, shuffle: bool = True, seed: int = 0):
        self.ds, self.max_frames, self.shuffle = ds, max_frames, shuffle
        self.rng = random.Random(seed)

    def _batches(self) -> list[list[int]]:
        order = sorted(range(len(self.ds)), key=self.ds.frames)
        batches, cur, longest = [], [], 0
        for i in order:
            f = self.ds.frames(i)
            if cur and max(longest, f) * (len(cur) + 1) > self.max_frames:
                batches.append(cur)
                cur, longest = [], 0
            cur.append(i)
            longest = max(longest, f)
        if cur:
            batches.append(cur)
        if self.shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())
