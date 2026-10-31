import json

import pytest
import torch
from fastapi.testclient import TestClient

from hushkey.config import Paths
from hushkey.preprocess import decode_wire_frame, preprocess

from .synth import wire_frames


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HUSHKEY_HOME", str(tmp_path / "home"))
    return Paths().ensure()


def make_manifest(d, n=6):
    d.mkdir(parents=True, exist_ok=True)
    texts = ["open the door", "send the email now"]
    with open(d / "manifest.jsonl", "w") as f:
        for i in range(n):
            hz = 2.0 if i % 2 == 0 else 4.5  # two visibly different "utterances"
            s = preprocess([decode_wire_frame(m) for m in wire_frames(1.2, fps=25, open_hz=hz)])
            s.save(d / f"{i}.npz", text=texts[i % 2])
            f.write(json.dumps({"path": f"{i}.npz", "text": texts[i % 2], "frames": len(s.rois)}) + "\n")
    return d / "manifest.jsonl"


def test_personalize_then_mwer(home, tmp_path):
    from hushkey.train import TrainConfig, train

    man = make_manifest(tmp_path / "enr")
    cfg = TrainConfig.for_stage("personalize", manifests=[str(man)], steps=30, lr=3e-3, warmup=5, val_fraction=0.0, tiny=True, workers=0, device="cpu", max_frames=200, log_every=5)
    import logging

    losses = []

    class Grab(logging.Handler):
        def emit(self, rec):
            if rec.getMessage().startswith("step") and "loss" in rec.getMessage():
                losses.append(float(rec.getMessage().split("loss ")[1].split()[0]))

    h = Grab()
    logging.getLogger("hushkey.train").addHandler(h)
    logging.getLogger("hushkey.train").setLevel(logging.INFO)
    hist = train(cfg)
    assert (home.adapters / "personal.pt").exists()
    assert len(losses) >= 5 and sum(losses[-2:]) < sum(losses[:2])

    cfg2 = TrainConfig.for_stage("mwer", manifests=[str(man)], steps=3, val_fraction=0.0, tiny=True, workers=0, device="cpu", nbest=3, mwer_beam=3)
    hist2 = train(cfg2)
    assert hist2["adapter"].endswith("personal.pt")
    blob = torch.load(home.adapters / "personal.pt", weights_only=True)
    assert blob["meta"]["stage"] == "mwer"


def test_server_stream_partial_final_and_enroll(home):
    from hushkey.engine import Engine, EngineConfig
    from hushkey.server import create_app

    engine = Engine(EngineConfig(tiny=True, beam_size=3, device="cpu"))
    client = TestClient(create_app(engine))
    assert client.get("/v1/health").json()["ok"]

    frames = wire_frames(1.6, fps=30)
    with client.websocket_connect("/v1/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "start", "app": "com.apple.TextEdit", "partials": False, "enroll_text": "hello world"})
        for f in frames:
            ws.send_json(f)
        ws.send_json({"type": "stop"})
        msg = ws.receive_json()
    assert msg["type"] == "final", msg
    assert msg["frames"] == 40 and "text" in msg and msg["latency_ms"] >= 0
    assert (home.enrollment / msg["enrolled"]).exists()
    line = json.loads((home.enrollment / "manifest.jsonl").read_text().splitlines()[0])
    assert line["text"] == "hello world"

    # no face at all → a clear error, not a crash
    blank = [dict(f, kp=None) for f in frames[:20]]
    r = client.post("/v1/transcribe", json={"frames": blank}).json()
    assert r["type"] == "error" and "face" in r["message"]


def test_cleanup_basic_format():
    from hushkey.cleanup import CleanupConfig, basic_format, finalize

    assert basic_format("I THINK I'LL BE LATE") == "I think I'll be late."
    assert basic_format("WHAT TIME IS IT") == "What time is it?"
    text, used = finalize("SEE YOU SOON", [], CleanupConfig(enabled=False))
    assert text == "See you soon." and not used
