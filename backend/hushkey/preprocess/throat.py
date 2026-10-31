"""Throat / submental motion from ordinary webcam pixels.

Silent mouthing has no vocal-fold vibration, but the tongue root, hyoid and larynx still
move: swallowing-like elevation on /k g ŋ/, jaw-opening muscles under the chin, etc.
We sample a face-aligned patch just below the chin and measure

    energy  – mean absolute frame difference (how much is moving)
    flow_y  – mean vertical Farnebäck optical flow (larynx/hyoid up vs down)
    valid   – 1 if the patch is inside the frame, else 0

This is a *weak* signal compared to the lips. It is used (a) as an extra channel that a
zero-initialised gate in the model may learn to use, and (b) by the app for hands-free
segmentation. ``hushkey eval --ablate-kinematics`` measures whether it helps on your data.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .keypoints import KP_INDEX, FaceFrame

ROI_W, ROI_H = 64, 40
#: Patch geometry in face units (inter-ocular distances), relative to the chin.
CENTER_BELOW_CHIN = 0.65
WIDTH_IOD = 1.3
HEIGHT_IOD = 0.8


def throat_patch(gray: np.ndarray, kp: np.ndarray) -> tuple[np.ndarray, bool]:
    """Face-aligned (ROI_H, ROI_W) patch below the chin, plus whether it lies inside the frame."""
    f = FaceFrame.from_keypoints(kp)
    chin_face = f.to_face(kp[KP_INDEX["chin"] : KP_INDEX["chin"] + 1])[0]
    c = chin_face + np.array([0.0, CENTER_BELOW_CHIN], np.float32)
    hw, hh = WIDTH_IOD / 2, HEIGHT_IOD / 2
    corners_face = np.array([[c[0] - hw, c[1] - hh], [c[0] + hw, c[1] - hh], [c[0] - hw, c[1] + hh]], np.float32)
    src = f.to_image(corners_face).astype(np.float32)
    dst = np.array([[0, 0], [ROI_W, 0], [0, ROI_H]], np.float32)
    M = cv2.getAffineTransform(src, dst)
    patch = cv2.warpAffine(gray, M, (ROI_W, ROI_H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    h, w = gray.shape[:2]
    full = f.to_image(np.array([[c[0] - hw, c[1] - hh], [c[0] + hw, c[1] + hh]], np.float32))
    inside = bool((full[:, 0] >= 0).all() and (full[:, 0] < w).all() and (full[:, 1] >= 0).all() and (full[:, 1] < h).all())
    return patch, inside


def throat_stream(frames: Sequence[np.ndarray], kps: Sequence[np.ndarray]) -> np.ndarray:
    """(T, 3) array of [energy, flow_y, valid]."""
    T = len(frames)
    out = np.zeros((T, 3), np.float32)
    prev = None
    for i, (g, k) in enumerate(zip(frames, kps)):
        patch, inside = throat_patch(g, k)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        out[i, 2] = 1.0 if inside else 0.0
        if prev is not None and inside:
            out[i, 0] = float(np.mean(np.abs(patch.astype(np.float32) - prev.astype(np.float32)))) / 255.0
            flow = cv2.calcOpticalFlowFarneback(prev, patch, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            out[i, 1] = float(np.mean(flow[..., 1])) / ROI_H
        prev = patch
    return out
