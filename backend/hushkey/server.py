"""Local inference server the macOS app talks to (127.0.0.1 only by default).

WebSocket protocol on ``/v1/stream`` (JSON text messages):

client → server
    {"type": "start", "app": "com.apple.mail", "partials": true, "enroll_text": null}
    {"type": "frame", "t": 12.345, "jpeg": "<base64 grayscale JPEG>", "kp": {"right_eye": [x, y], ...} | null}
    {"type": "stop"}            → final result
    {"type": "cancel"}          → discard utterance

server → client
    {"type": "ready", "info": {...}}
    {"type": "partial", "text": "..."}                     (CTC-greedy, ~1/s while recording)
    {"type": "final", "text": "...", "raw": "...", "alternatives": [...], "latency_ms": 840, ...}
    {"type": "error", "message": "..."}

If ``enroll_text`` is set on ``start``, the preprocessed utterance is also saved with that
transcript to the enrollment folder for ``hushkey train personalize``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .engine import Engine
from .preprocess import NoFaceError, RawFrame, decode_wire_frame, preprocess

log = logging.getLogger("hushkey.server")

MAX_SECONDS = 30.0
PARTIAL_EVERY_S = 1.0


class TranscribeRequest(BaseModel):
    frames: list[dict]
    app: Optional[str] = None
    enroll_text: Optional[str] = None


def save_enrollment(engine: Engine, sample, text: str) -> str:
    d = engine.paths.enrollment
    d.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.npz"
    sample.save(d / name, text=text)
    with open(d / "manifest.jsonl", "a") as f:
        f.write(json.dumps({"path": name, "text": text, "frames": int(len(sample.rois))}) + "\n")
    return name


def run_final(engine: Engine, frames: list[RawFrame], app: Optional[str], enroll_text: Optional[str]) -> dict:
    t0 = time.time()
    try:
        sample = preprocess(frames)
    except NoFaceError as e:
        return {"type": "error", "message": str(e)}
    pre_ms = int((time.time() - t0) * 1000)
    out = engine.transcribe(sample, app=app)
    out["preprocess_ms"] = pre_ms
    out["latency_ms"] = out.get("latency_ms", 0) + pre_ms
    if enroll_text:
        out["enrolled"] = save_enrollment(engine, sample, enroll_text)
    return {"type": "final", **out}


def run_partial(engine: Engine, frames: list[RawFrame]) -> Optional[str]:
    try:
        return engine.partial(preprocess(frames))
    except NoFaceError:
        return None


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="Hushkey", version="0.1.0")

    @app.get("/v1/health")
    def health():
        return {"ok": True, **engine.info()}

    @app.post("/v1/transcribe")
    async def transcribe(req: TranscribeRequest):
        frames = [decode_wire_frame(f) for f in req.frames]
        return await asyncio.to_thread(run_final, engine, frames, req.app, req.enroll_text)

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"type": "ready", "info": engine.info()})
        frames: list[RawFrame] = []
        recording = False
        opts: dict = {}
        partial_task: Optional[asyncio.Task] = None
        last_partial = 0.0

        async def do_partial(snapshot: list[RawFrame]):
            text = await asyncio.to_thread(run_partial, engine, snapshot)
            if text is not None and recording:
                await ws.send_json({"type": "partial", "text": text})

        try:
            while True:
                msg = json.loads(await ws.receive_text())
                kind = msg.get("type")
                if kind == "start":
                    frames, recording, opts = [], True, msg
                    last_partial = time.time()
                elif kind == "frame" and recording:
                    try:
                        frames.append(decode_wire_frame(msg))
                    except Exception as e:  # malformed frame: skip, don't kill the session
                        log.warning("bad frame: %s", e)
                        continue
                    if frames[-1].t - frames[0].t > MAX_SECONDS:
                        frames.pop(0)
                    now = time.time()
                    if opts.get("partials", True) and now - last_partial > PARTIAL_EVERY_S and (partial_task is None or partial_task.done()):
                        last_partial = now
                        partial_task = asyncio.create_task(do_partial(list(frames)))
                elif kind == "stop" and recording:
                    recording = False
                    snapshot, frames = frames, []
                    result = await asyncio.to_thread(run_final, engine, snapshot, opts.get("app"), opts.get("enroll_text"))
                    await ws.send_json(result)
                elif kind == "cancel":
                    recording, frames = False, []
                elif kind == "ping":
                    await ws.send_json({"type": "pong"})
        except WebSocketDisconnect:
            return

    return app
