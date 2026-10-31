"""Landmark-source-agnostic face keypoints and the kinematic feature stream.

The macOS app (Apple Vision) and the training-data preparation (MediaPipe FaceMesh)
both reduce their landmarks to the same 11 named points, in *pixel coordinates of the
frame that is sent alongside them* (origin top-left, x right, y down):

    right_eye, left_eye     eye centres (we re-sort by x, so mirroring can't swap them)
    nose_tip
    mouth_center
    lip_upper_inner, lip_lower_inner
    lip_upper_outer, lip_lower_outer
    lip_left, lip_right     mouth corners
    chin                    lowest point of the jaw contour

From these we derive a per-frame kinematic vector. Everything is expressed in a
face-aligned frame (eye line horizontal) and scaled by inter-ocular distance, then
normalised per utterance. That makes the stream comparable across landmark sources,
camera distances and head roll.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

KEYPOINT_NAMES: tuple[str, ...] = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_center",
    "lip_upper_inner",
    "lip_lower_inner",
    "lip_upper_outer",
    "lip_lower_outer",
    "lip_left",
    "lip_right",
    "chin",
)
KP_INDEX = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

#: Points that Auto-AVSR's similarity warp is fitted on (in this order).
STABLE_NAMES = ("right_eye", "left_eye", "nose_tip", "mouth_center")

GEOM_NAMES = ("aperture_inner", "aperture_outer", "width", "jaw", "mouth_drop", "roundness")
#: Final kinematic channels fed to the model.
KIN_NAMES: tuple[str, ...] = (
    *GEOM_NAMES,
    *(f"d_{n}" for n in GEOM_NAMES),
    "throat_energy",
    "throat_flow_y",
    "head_motion",
    "throat_valid",
)
KIN_DIM = len(KIN_NAMES)  # 16


def keypoints_from_mapping(kp: Optional[Mapping[str, Sequence[float]]]) -> Optional[np.ndarray]:
    """Convert a ``{name: [x, y]}`` mapping (wire format) to an (11, 2) array."""
    if not kp:
        return None
    try:
        arr = np.array([kp[n] for n in KEYPOINT_NAMES], dtype=np.float32)
    except KeyError:
        return None
    if arr.shape != (len(KEYPOINT_NAMES), 2) or not np.isfinite(arr).all():
        return None
    return canonicalize(arr)


def canonicalize(arr: np.ndarray) -> np.ndarray:
    """Order eye/corner pairs by image x so mirrored input can't swap them.

    Auto-AVSR's mean face puts the subject's right eye on the image left (x smaller),
    which is what an un-mirrored webcam frame shows.
    """
    arr = arr.copy()
    re, le = KP_INDEX["right_eye"], KP_INDEX["left_eye"]
    if arr[re, 0] > arr[le, 0]:
        arr[[re, le]] = arr[[le, re]]
    lr, ll = KP_INDEX["lip_right"], KP_INDEX["lip_left"]
    if arr[lr, 0] > arr[ll, 0]:
        arr[[lr, ll]] = arr[[ll, lr]]
    return arr


def stable_points(arr: np.ndarray) -> np.ndarray:
    """(4, 2) points for the similarity warp: right eye, left eye, nose tip, mouth centre."""
    return arr[[KP_INDEX[n] for n in STABLE_NAMES]]


@dataclass
class FaceFrame:
    """Rotation/scale that maps image pixels into a face-aligned, IOD-normalised frame."""

    origin: np.ndarray  # eye midpoint, (2,)
    rot: np.ndarray  # (2, 2), rows = face x-axis, face y-axis (y points down the face)
    iod: float

    @classmethod
    def from_keypoints(cls, arr: np.ndarray) -> "FaceFrame":
        re, le = arr[KP_INDEX["right_eye"]], arr[KP_INDEX["left_eye"]]
        axis = le - re
        iod = float(np.linalg.norm(axis))
        if iod < 1e-3:
            axis, iod = np.array([1.0, 0.0], np.float32), 1.0
        ex = axis / iod
        ey = np.array([-ex[1], ex[0]], dtype=np.float32)  # rotate +90°: image-down when upright
        return cls(origin=(re + le) / 2.0, rot=np.stack([ex, ey]).astype(np.float32), iod=iod)

    def to_face(self, pts: np.ndarray) -> np.ndarray:
        return ((pts - self.origin) @ self.rot.T) / self.iod

    def to_image(self, pts_face: np.ndarray) -> np.ndarray:
        return (pts_face * self.iod) @ self.rot + self.origin


def geometry_features(arr: np.ndarray) -> np.ndarray:
    """Six articulatory measurements for one frame (face-aligned, IOD units)."""
    f = FaceFrame.from_keypoints(arr)
    p = f.to_face(arr)
    g = lambda n: p[KP_INDEX[n]]  # noqa: E731
    ap_in = float(np.linalg.norm(g("lip_lower_inner") - g("lip_upper_inner")))
    ap_out = float(np.linalg.norm(g("lip_lower_outer") - g("lip_upper_outer")))
    width = float(np.linalg.norm(g("lip_left") - g("lip_right")))
    jaw = float(np.linalg.norm(g("chin") - g("nose_tip")))
    mouth_drop = float(g("mouth_center")[1] - g("nose_tip")[1])
    roundness = ap_out / (width + 1e-3)
    return np.array([ap_in, ap_out, width, jaw, mouth_drop, roundness], dtype=np.float32)


def interpolate_missing(kps: list[Optional[np.ndarray]]) -> Optional[list[np.ndarray]]:
    """Linearly fill frames without a face; hold the first/last detection at the ends."""
    valid = [i for i, k in enumerate(kps) if k is not None]
    if not valid:
        return None
    out: list[np.ndarray] = [None] * len(kps)  # type: ignore[list-item]
    for i in valid:
        out[i] = kps[i]
    for a, b in zip(valid[:-1], valid[1:]):
        for j in range(a + 1, b):
            w = (j - a) / (b - a)
            out[j] = (1 - w) * kps[a] + w * kps[b]
    for j in range(valid[0]):
        out[j] = kps[valid[0]]
    for j in range(valid[-1] + 1, len(kps)):
        out[j] = kps[valid[-1]]
    return out


def kinematic_stream(
    kps: Sequence[np.ndarray],
    throat: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Build the (T, KIN_DIM) kinematic stream.

    Args:
        kps: per-frame (11, 2) keypoints, already gap-filled.
        throat: optional (T, 3) array of [energy, flow_y, valid] from ``throat.py``.
        normalize: per-utterance z-normalisation of every channel except ``throat_valid``.
    """
    T = len(kps)
    geom = np.stack([geometry_features(k) for k in kps]) if T else np.zeros((0, 6), np.float32)
    d_geom = np.vstack([np.zeros((1, 6), np.float32), np.diff(geom, axis=0)]) if T else geom

    nose = np.stack([k[KP_INDEX["nose_tip"]] for k in kps]) if T else np.zeros((0, 2))
    iod = np.array([FaceFrame.from_keypoints(k).iod for k in kps]) if T else np.zeros(0)
    head = np.zeros(T, np.float32)
    if T > 1:
        head[1:] = np.linalg.norm(np.diff(nose, axis=0), axis=1) / np.maximum(iod[1:], 1e-3)

    if throat is None:
        throat = np.zeros((T, 3), np.float32)
    kin = np.concatenate([geom, d_geom, throat[:, :2], head[:, None], throat[:, 2:3]], axis=1).astype(np.float32)

    if normalize and T > 0:
        body = kin[:, :-1]
        mu = np.median(body, axis=0, keepdims=True)
        sd = body.std(axis=0, keepdims=True)
        kin[:, :-1] = (body - mu) / np.maximum(sd, 1e-2)
        kin[:, :-1] = np.clip(kin[:, :-1], -8.0, 8.0)
    return kin
