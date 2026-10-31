"""Convert raw talking-face videos + transcripts into Hushkey ``.npz`` utterances.

Supported layouts
-----------------
* ``lrs3``  – the official LRS3 tree: ``<root>/<split>/<speaker>/<id>.mp4`` with a matching
              ``<id>.txt`` whose first line is ``Text:  THE TRANSCRIPT``.
* ``csv``   – any corpus described by a CSV with columns ``path,text`` (paths relative to the CSV).

Each video is decoded, MediaPipe Face Landmarker gives the 11 keypoints per frame, and the exact
same ``preprocess()`` used by the live server produces mouth ROIs + the kinematic stream.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator

import cv2

from ..preprocess import NoFaceError, RawFrame, preprocess
from .facemesh import FaceMeshKeypoints


def iter_lrs3(root: Path) -> Iterator[tuple[Path, str]]:
    for mp4 in sorted(root.rglob("*.mp4")):
        txt = mp4.with_suffix(".txt")
        if not txt.exists():
            continue
        first = txt.read_text(errors="ignore").splitlines()[0]
        text = first.split(":", 1)[1].strip() if first.lower().startswith("text") else first.strip()
        yield mp4, text


def iter_csv(csv_path: Path) -> Iterator[tuple[Path, str]]:
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            p = Path(row["path"])
            yield (p if p.is_absolute() else csv_path.parent / p), row["text"]


def read_video(path: Path, kp: FaceMeshKeypoints) -> list[RawFrame]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    kp.reset(fps)
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(RawFrame(t=i / fps, gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), kp=kp(rgb)))
        i += 1
    cap.release()
    return frames


def prepare(items: Iterable[tuple[Path, str]], out_dir: Path, base: Path, limit: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    kp = FaceMeshKeypoints()
    stats = {"ok": 0, "no_face": 0, "failed": 0}
    with open(out_dir / "manifest.jsonl", "a") as man:
        for n, (video, text) in enumerate(items):
            if limit and n >= limit:
                break
            try:
                rel = video.relative_to(base).with_suffix(".npz")
            except ValueError:
                rel = Path(video.stem + ".npz")
            dst = out_dir / rel
            if dst.exists():
                continue
            try:
                sample = preprocess(read_video(video, kp))
            except NoFaceError:
                stats["no_face"] += 1
                continue
            except Exception as e:  # corrupt video etc.
                print(f"! {video}: {e}")
                stats["failed"] += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            sample.save(dst, text=text)
            man.write(json.dumps({"path": str(rel), "text": text, "frames": int(len(sample.rois))}) + "\n")
            stats["ok"] += 1
            if stats["ok"] % 100 == 0:
                print(stats)
    kp.close()
    return stats
