"""Keypoints for offline video (training-data prep) with MediaPipe Face Landmarker.

Maps the 478-point face mesh onto the same 11 named keypoints the macOS app computes with
Apple Vision, so preprocessing is identical in training and at inference.
Install with ``pip install 'hushkey[prepare]'``. Uses the MediaPipe *Tasks* API (the legacy
``mp.solutions`` API no longer ships in current MediaPipe releases); the model file is
downloaded on first use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import Paths
from ..preprocess.keypoints import KEYPOINT_NAMES

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Face-mesh indices (subject's right = image left in an un-mirrored frame)
_MESH = {
    "right_eye": [33, 133, 159, 145],
    "left_eye": [362, 263, 386, 374],
    # Apple Vision's `nose` region outlines the lower nose; its centroid sits between the tip
    # and the nostrils, so we average the same area here to keep both sources consistent.
    "nose_tip": [1, 2, 98, 327],
    "mouth_center": [0, 17, 61, 291],
    "lip_upper_inner": [13],
    "lip_lower_inner": [14],
    "lip_upper_outer": [0],
    "lip_lower_outer": [17],
    "lip_left": [291],
    "lip_right": [61],
    "chin": [152],
}


def mesh_to_keypoints(pts: np.ndarray) -> np.ndarray:
    """(≥468, 2) pixel mesh → (11, 2) Hushkey keypoints."""
    return np.stack([pts[_MESH[n]].mean(axis=0) for n in KEYPOINT_NAMES]).astype(np.float32)


def ensure_model(path: Optional[Path] = None) -> Path:
    path = path or (Paths().ensure().models / "face_landmarker.task")
    if not path.exists():
        print(f"↓ {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class FaceMeshKeypoints:
    """Call once per frame, in order (VIDEO running mode keeps tracking state)."""

    def __init__(self, model_path: Optional[Path] = None, fps: float = 25.0):
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as e:  # pragma: no cover
            raise ImportError("MediaPipe is needed for offline video preparation: pip install 'hushkey[prepare]'") from e
        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.dt_ms = 1000.0 / fps
        self.ts = 0.0

    def reset(self, fps: float = 25.0):
        # timestamps must increase monotonically across videos too
        self.ts += 1000.0
        self.dt_ms = 1000.0 / fps

    def __call__(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.lm.detect_for_video(img, int(self.ts))
        self.ts += self.dt_ms
        if not res.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float32)
        return mesh_to_keypoints(pts)

    def close(self):
        self.lm.close()
