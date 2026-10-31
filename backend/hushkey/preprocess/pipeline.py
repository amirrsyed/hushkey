"""One preprocessing path for live inference, enrollment and training-data preparation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from ..config import MODEL_FPS
from .keypoints import canonicalize, interpolate_missing, keypoints_from_mapping, kinematic_stream, stable_points
from .mouth import mouth_rois
from .throat import throat_stream


class NoFaceError(ValueError):
    """Raised when no frame in an utterance had a usable face."""


@dataclass
class RawFrame:
    t: float  # seconds
    gray: np.ndarray  # (H, W) uint8
    kp: Optional[np.ndarray]  # (11, 2) or None


@dataclass
class Sample:
    """A preprocessed utterance: what the model and the training sets consume."""

    rois: np.ndarray  # (T, 96, 96) uint8 mouth patches at 25 fps
    kin: np.ndarray  # (T, KIN_DIM) float32
    face_ratio: float  # fraction of original frames that had a detected face

    def save(self, path, text: str = "") -> None:
        np.savez_compressed(path, rois=self.rois, kin=self.kin, face_ratio=self.face_ratio, text=np.array(text))

    @staticmethod
    def load(path) -> tuple["Sample", str]:
        z = np.load(path, allow_pickle=False)
        return Sample(z["rois"], z["kin"].astype(np.float32), float(z["face_ratio"])), str(z["text"])


def decode_wire_frame(msg: Mapping) -> RawFrame:
    """Decode one ``{"t", "jpeg" (base64), "kp"}`` frame message from the app."""
    buf = np.frombuffer(base64.b64decode(msg["jpeg"]), dtype=np.uint8)
    gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("could not decode JPEG frame")
    return RawFrame(t=float(msg["t"]), gray=gray, kp=keypoints_from_mapping(msg.get("kp")))


def resample_indices(times: Sequence[float], fps: float = MODEL_FPS) -> list[int]:
    """Nearest-frame indices that resample a variable-rate capture to a fixed rate."""
    if len(times) == 0:
        return []
    t = np.asarray(times, dtype=np.float64)
    t = t - t[0]
    n = max(1, int(round(t[-1] * fps)) + 1)
    grid = np.arange(n) / fps
    idx = np.searchsorted(t, grid)
    idx = np.clip(idx, 0, len(t) - 1)
    prev = np.clip(idx - 1, 0, len(t) - 1)
    choose_prev = np.abs(t[prev] - grid) <= np.abs(t[idx] - grid)
    return np.where(choose_prev, prev, idx).tolist()


def preprocess(frames: Sequence[RawFrame], fps: float = MODEL_FPS, resample: bool = True) -> Sample:
    if not frames:
        raise NoFaceError("empty utterance")
    face_ratio = sum(f.kp is not None for f in frames) / len(frames)
    idx = resample_indices([f.t for f in frames], fps) if resample else list(range(len(frames)))
    picked = [frames[i] for i in idx]
    kps = interpolate_missing([None if f.kp is None else canonicalize(f.kp) for f in picked])
    if kps is None:
        raise NoFaceError("no face found in any frame — check lighting and framing")
    grays = [f.gray for f in picked]
    rois = mouth_rois(grays, [stable_points(k) for k in kps])
    kin = kinematic_stream(kps, throat_stream(grays, kps))
    return Sample(rois=rois, kin=kin, face_ratio=face_ratio)
