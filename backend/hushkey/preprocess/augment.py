"""Training augmentations that push voiced-speech corpora (LRS3, VoxCeleb2) toward
*silent, webcam* speech.

Petridis et al. (ICASSP 2018) found lip-reading accuracy drops by up to ~8.5 points
when a model trained on normal speech is tested on silent speech: people articulate
less and more slowly without auditory feedback. We can't buy a large silent corpus, so
we simulate its main differences on the voiced data we have:

* ``attenuate``  – hypo-articulation: shrink each frame's deviation from a slow moving
                   average (v' = m + α·(v − m)), applied identically to the geometry
                   channels of the kinematic stream.
* ``tempo``      – silent speech is slower: resample time by 0.8–1.25×.
* ``webcam``     – JPEG re-compression (Chaplin streams at quality 25), blur, gamma,
                   sensor noise.
* ``time_mask``  – Auto-AVSR's adaptive time masking.
* kinematic dropout – zero the throat channels so the model never depends on them.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

from .keypoints import KIN_NAMES
from .mouth import CROP, PATCH, PIXEL_MEAN, PIXEL_STD

_GEOM = [i for i, n in enumerate(KIN_NAMES) if not n.startswith(("d_", "throat", "head"))]
_DGEOM = [i for i, n in enumerate(KIN_NAMES) if n.startswith("d_")]
_THROAT = [i for i, n in enumerate(KIN_NAMES) if n.startswith("throat")]


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], 0)
    c = np.cumsum(xp, axis=0, dtype=np.float64)
    c = np.concatenate([np.zeros_like(c[:1]), c], 0)
    return ((c[k:] - c[:-k]) / k).astype(np.float32)


def attenuate(rois: np.ndarray, kin: np.ndarray, alpha: float, window: int = 9) -> tuple[np.ndarray, np.ndarray]:
    v = rois.astype(np.float32)
    m = _moving_average(v, window)
    v = np.clip(m + alpha * (v - m), 0, 255).astype(np.uint8)
    kin = kin.copy()
    g = kin[:, _GEOM]
    gm = _moving_average(g, window)
    kin[:, _GEOM] = gm + alpha * (g - gm)
    kin[:, _DGEOM] *= alpha
    return v, kin


def tempo(rois: np.ndarray, kin: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """factor > 1 → slower (more frames)."""
    T = len(rois)
    n = max(2, int(round(T * factor)))
    idx = np.clip(np.round(np.linspace(0, T - 1, n)).astype(int), 0, T - 1)
    return rois[idx], kin[idx]


def webcam(rois: np.ndarray, rng: random.Random) -> np.ndarray:
    q = rng.randint(20, 70)
    gamma = rng.uniform(0.7, 1.4)
    blur = rng.random() < 0.3
    noise = rng.uniform(0, 6)
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
    out = np.empty_like(rois)
    for i, f in enumerate(rois):
        f = cv2.LUT(f, lut)
        if blur:
            f = cv2.GaussianBlur(f, (3, 3), 0)
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])
        f = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else f
        out[i] = f
    if noise > 0:
        out = np.clip(out.astype(np.float32) + np.random.default_rng(rng.randint(0, 2**31)).normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    return out


def time_mask(x: np.ndarray, rng: random.Random, window: int = 10, stride: int = 25) -> np.ndarray:
    """Auto-AVSR AdaptiveTimeMask: one mask of up to `window` frames per `stride` frames, filled with the mean."""
    x = x.copy()
    n = len(x) // stride
    for _ in range(n):
        w = rng.randint(0, window)
        if w == 0 or len(x) - w <= 0:
            continue
        s = rng.randint(0, len(x) - w)
        x[s : s + w] = x.mean(axis=0)
    return x


class SilentSpeechAugment:
    def __init__(self, p_attenuate=0.7, alpha=(0.45, 0.9), p_tempo=0.5, tempo_range=(0.8, 1.25), p_webcam=0.8, p_kin_drop=0.3, seed=None):
        self.p_attenuate, self.alpha = p_attenuate, alpha
        self.p_tempo, self.tempo_range = p_tempo, tempo_range
        self.p_webcam, self.p_kin_drop = p_webcam, p_kin_drop
        self.rng = random.Random(seed)

    def __call__(self, rois: np.ndarray, kin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.rng
        if r.random() < self.p_attenuate:
            rois, kin = attenuate(rois, kin, r.uniform(*self.alpha))
        if r.random() < self.p_tempo:
            rois, kin = tempo(rois, kin, r.uniform(*self.tempo_range))
        if r.random() < self.p_webcam:
            rois = webcam(rois, r)
        kin = kin.copy()
        if r.random() < self.p_kin_drop:
            kin[:, _THROAT] = 0.0
        # random 88×88 crop (Auto-AVSR train transform), then normalise
        ox, oy = r.randint(0, PATCH - CROP), r.randint(0, PATCH - CROP)
        x = rois[:, oy : oy + CROP, ox : ox + CROP].astype(np.float32) / 255.0
        x = (x - PIXEL_MEAN) / PIXEL_STD
        x = time_mask(x, r)
        return x.astype(np.float32), kin.astype(np.float32)
