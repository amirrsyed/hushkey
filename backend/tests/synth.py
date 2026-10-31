"""Synthetic talking faces with exact keypoints, so tests need no dataset or camera."""

from __future__ import annotations

import base64
import math

import cv2
import numpy as np

from hushkey.preprocess.keypoints import KEYPOINT_NAMES


def face_frame(t: float, W: int = 320, H: int = 360, cx: float = 160, cy: float = 150, iod: float = 70, roll_deg: float = 0.0, open_hz: float = 3.0, throat_hz: float = 2.0):
    """One grayscale frame + its 11 keypoints. Mouth opens/closes at `open_hz`; a 'larynx' blob bobs at `throat_hz`."""
    img = np.full((H, W), 60, np.uint8)
    a = math.radians(roll_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def P(x, y):  # face units (iod) → pixels
        v = R @ np.array([x, y]) * iod + np.array([cx, cy])
        return v.astype(np.float32)

    opening = 0.08 + 0.12 * (0.5 + 0.5 * math.sin(2 * math.pi * open_hz * t))
    kp = {
        "right_eye": P(-0.5, 0.0),
        "left_eye": P(0.5, 0.0),
        "nose_tip": P(0.0, 0.7),
        "mouth_center": P(0.0, 1.15),
        "lip_upper_inner": P(0.0, 1.15 - opening / 2),
        "lip_lower_inner": P(0.0, 1.15 + opening / 2),
        "lip_upper_outer": P(0.0, 1.15 - opening / 2 - 0.08),
        "lip_lower_outer": P(0.0, 1.15 + opening / 2 + 0.08),
        "lip_left": P(0.45, 1.15),
        "lip_right": P(-0.45, 1.15),
        "chin": P(0.0, 1.7),
    }
    c = tuple(int(v) for v in P(0, 0.8))
    cv2.ellipse(img, c, (int(0.95 * iod), int(1.25 * iod)), roll_deg, 0, 360, 170, -1)
    for e in ("right_eye", "left_eye"):
        cv2.circle(img, tuple(int(v) for v in kp[e]), int(0.1 * iod), 20, -1)
    cv2.circle(img, tuple(int(v) for v in kp["nose_tip"]), int(0.06 * iod), 110, -1)
    cv2.ellipse(img, tuple(int(v) for v in kp["mouth_center"]), (int(0.42 * iod), max(1, int(opening / 2 * iod) + 2)), roll_deg, 0, 360, 30, -1)
    # neck + larynx blob
    neck = tuple(int(v) for v in P(0, 2.3))
    cv2.rectangle(img, (neck[0] - int(0.5 * iod), int(P(0, 1.9)[1])), (neck[0] + int(0.5 * iod), H), 150, -1)
    ly = P(0, 2.3 + 0.12 * math.sin(2 * math.pi * throat_hz * t))
    cv2.circle(img, tuple(int(v) for v in ly), int(0.12 * iod), 100, -1)
    arr = np.stack([kp[n] for n in KEYPOINT_NAMES])
    return img, arr


def wire_frames(seconds: float = 2.0, fps: float = 30.0, **kw) -> list[dict]:
    out = []
    for i in range(int(seconds * fps)):
        t = i / fps
        img, kp = face_frame(t, **kw)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out.append({"type": "frame", "t": 100.0 + t, "jpeg": base64.b64encode(buf.tobytes()).decode(), "kp": {n: kp[j].tolist() for j, n in enumerate(KEYPOINT_NAMES)}})
    return out
