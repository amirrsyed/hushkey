# Adapted from Auto-AVSR / Chaplin `video_process.py`
# Copyright 2023 Imperial College London (Pingchuan Ma), Apache 2.0.
# Changes for Hushkey: takes our 4 stable keypoints directly (no detector), never raises
# on off-centre faces (clamps instead), works on grayscale frames of any size.
"""Mouth-region-of-interest extraction matching the Auto-AVSR training pipeline.

Each frame is similarity-warped so the eyes / nose tip / mouth centre land on the
mean-face reference in a 256×256 canvas, then a 96×96 patch centred on the mouth is cut.
At model input, a 88×88 centre crop is normalised with mean 0.421 / std 0.165.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from ..config import MEAN_FACE_FILE

CANVAS = 256
PATCH = 96
CROP = 88
PIXEL_MEAN = 0.421
PIXEL_STD = 0.165


def _stable_reference() -> np.ndarray:
    ref = np.load(MEAN_FACE_FILE)  # 68-point iBUG mean face in a 256×256 canvas
    return np.vstack(
        [
            ref[36:42].mean(axis=0),  # right eye (image left)
            ref[42:48].mean(axis=0),  # left eye
            ref[31:36].mean(axis=0),  # nose
            ref[48:68].mean(axis=0),  # mouth
        ]
    ).astype(np.float32)


_REFERENCE = None


def stable_reference() -> np.ndarray:
    global _REFERENCE
    if _REFERENCE is None:
        _REFERENCE = _stable_reference()
    return _REFERENCE


def smooth_landmarks(stable: Sequence[np.ndarray], window_margin: int = 12) -> list[np.ndarray]:
    """Temporal smoothing used by Auto-AVSR: average over ±window/2 frames, keep the frame's centroid."""
    out = []
    n = len(stable)
    for i in range(n):
        m = min(window_margin // 2, i, n - 1 - i)
        sm = np.mean([stable[j] for j in range(i - m, i + m + 1)], axis=0)
        sm = sm + (stable[i].mean(axis=0) - sm.mean(axis=0))
        out.append(sm.astype(np.float32))
    return out


def warp_frame(gray: np.ndarray, stable: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Similarity-warp one frame onto the reference. Returns (canvas, warped points, 2×3 matrix)."""
    ref = stable_reference()
    M, _ = cv2.estimateAffinePartial2D(stable.astype(np.float32), ref, method=cv2.LMEDS)
    if M is None:  # degenerate fit: fall back to a pure scale+translate on the eye line
        M, _ = cv2.estimateAffinePartial2D(stable[:2].astype(np.float32), ref[:2], method=0)
        if M is None:
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    canvas = cv2.warpAffine(gray, M, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pts = stable @ M[:, :2].T + M[:, 2]
    return canvas, pts.astype(np.float32), M


def cut_patch(canvas: np.ndarray, center: np.ndarray, size: int = PATCH) -> np.ndarray:
    half = size // 2
    cx = int(round(np.clip(center[0], half, canvas.shape[1] - half)))
    cy = int(round(np.clip(center[1], half, canvas.shape[0] - half)))
    return canvas[cy - half : cy + half, cx - half : cx + half].copy()


def mouth_rois(frames: Sequence[np.ndarray], stable: Sequence[np.ndarray]) -> np.ndarray:
    """(T, 96, 96) uint8 mouth patches from grayscale frames and gap-filled stable points."""
    smoothed = smooth_landmarks(stable)
    out = np.empty((len(frames), PATCH, PATCH), dtype=np.uint8)
    for i, (f, s) in enumerate(zip(frames, smoothed)):
        canvas, pts, _ = warp_frame(f, s)
        out[i] = cut_patch(canvas, pts[3])
    return out


def to_model_input(rois: np.ndarray, center_crop: bool = True) -> np.ndarray:
    """(T, 96, 96) uint8 → (T, 88, 88) float32, normalised as in Auto-AVSR."""
    x = rois.astype(np.float32) / 255.0
    if center_crop:
        o = (PATCH - CROP) // 2
        x = x[:, o : o + CROP, o : o + CROP]
    return (x - PIXEL_MEAN) / PIXEL_STD
