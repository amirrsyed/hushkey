"""``hushkey`` command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

from .config import Paths

HF = "https://huggingface.co"
#: Auto-AVSR VSR model + LRS3 subword RNN-LM, as mirrored by Chaplin (amanvirparhar/chaplin).
DOWNLOADS = {
    "LRS3_V_WER19.1/model.pth": f"{HF}/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth",
    "LRS3_V_WER19.1/model.json": f"{HF}/Amanvir/LRS3_V_WER19.1/resolve/main/model.json",
    "lm_en_subword/model.pth": f"{HF}/Amanvir/lm_en_subword/resolve/main/model.pth",
    "lm_en_subword/model.json": f"{HF}/Amanvir/lm_en_subword/resolve/main/model.json",
}


def cmd_download(a):
    paths = Paths().ensure()
    for rel, url in DOWNLOADS.items():
        dst = paths.models / rel
        if dst.exists() and not a.force:
            print(f"✓ {rel} (exists)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        print(f"↓ {rel}")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        tmp.rename(dst)
    print(f"models in {paths.models}")


def _engine_cfg(a):
    from .cleanup import CleanupConfig
    from .engine import EngineConfig

    vocab = []
    if getattr(a, "vocabulary", None):
        vocab = [w.strip() for w in Path(a.vocabulary).read_text().splitlines() if w.strip()]
    return EngineConfig(
        device=a.device,
        beam_size=a.beam,
        use_lm=not a.no_lm,
        adapter=a.adapter,
        tiny=a.tiny,
        cleanup=CleanupConfig(enabled=a.llm, model=a.llm_model, vocabulary=vocab),
    )


def cmd_serve(a):
    import uvicorn

    from .engine import Engine
    from .server import create_app

    engine = Engine(_engine_cfg(a))
    print(json.dumps(engine.info()))
    uvicorn.run(create_app(engine), host=a.host, port=a.port, log_level="info", ws_max_size=32 * 1024 * 1024)


def cmd_train(a):
    from .train import TrainConfig, train

    manifests = a.manifest or []
    if a.stage in ("personalize", "mwer") and not manifests:
        enr = Paths().enrollment / "manifest.jsonl"
        if not enr.exists():
            sys.exit(f"No enrollment data at {enr}. Record some in the app (Hushkey ▸ Train on my mouth…).")
        manifests = [str(enr)]
    cfg = TrainConfig.for_stage(
        a.stage,
        manifests=manifests,
        replay_manifests=a.replay or [],
        out=a.out,
        init_adapter=a.init_adapter,
        steps=a.steps,
        lr=a.lr,
        max_frames=a.max_frames,
        lora_rank=a.lora_rank,
        use_kinematics=not a.no_kinematics,
        device=a.device,
        tiny=a.tiny,
        workers=a.workers,
    )
    hist = train(cfg)
    print(json.dumps({k: hist[k] for k in ("best_wer", "adapter") if k in hist}, indent=2))


def cmd_eval(a):
    from .evaluate import run_eval

    r = run_eval(a.manifest, a.adapter, a.beam, not a.no_lm, a.tiny, a.device, a.limit, a.compare_adapter, a.compare_zero_kinematics)
    print(json.dumps(r, indent=2))


def cmd_prepare(a):
    from .data.prepare import iter_csv, iter_lrs3, prepare

    src = Path(a.source)
    items = iter_lrs3(src) if a.layout == "lrs3" else iter_csv(src)
    base = src if src.is_dir() else src.parent
    print(prepare(items, Path(a.out), base, limit=a.limit))


def main(argv=None):
    p = argparse.ArgumentParser(prog="hushkey", description="Silent-speech webcam dictation backend")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch pretrained Auto-AVSR VSR weights + LM")
    d.add_argument("--force", action="store_true")
    d.set_defaults(fn=cmd_download)

    def runtime(sp):
        sp.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
        sp.add_argument("--adapter", default=None, help="adapter .pt (default: personal.pt, else silent.pt, else none)")
        sp.add_argument("--beam", type=int, default=10)
        sp.add_argument("--no-lm", action="store_true", help="disable the RNN language model")
        sp.add_argument("--tiny", action="store_true", help="random tiny model (UI dev / CI, outputs gibberish)")

    s = sub.add_parser("serve", help="run the local server the Mac app connects to")
    runtime(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--llm", action="store_true", help="LLM cleanup through a local Ollama server")
    s.add_argument("--llm-model", default="qwen3:4b")
    s.add_argument("--vocabulary", default=None, help="text file, one personal word/name per line")
    s.set_defaults(fn=cmd_serve)

    t = sub.add_parser("train", help="train adapters (adapt | personalize | mwer)")
    t.add_argument("stage", choices=["adapt", "personalize", "mwer"])
    t.add_argument("--manifest", action="append", help="manifest.jsonl (repeatable); personalize/mwer default to enrollment data")
    t.add_argument("--replay", action="append", help="public-data manifest mixed in during personalize")
    t.add_argument("--out")
    t.add_argument("--init-adapter")
    t.add_argument("--steps", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--max-frames", type=int)
    t.add_argument("--lora-rank", type=int, default=8)
    t.add_argument("--no-kinematics", action="store_true")
    t.add_argument("--device", default="auto")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--tiny", action="store_true")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="WER on a manifest, with optional paired comparison")
    runtime(e)
    e.add_argument("manifest")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--compare-adapter", default=None, help="baseline adapter path, or 'none' for the plain base model")
    e.add_argument("--compare-zero-kinematics", action="store_true")
    e.set_defaults(fn=cmd_eval)

    pr = sub.add_parser("prepare", help="convert videos+transcripts to training .npz (needs mediapipe)")
    pr.add_argument("source", help="LRS3 root dir, or a CSV with path,text")
    pr.add_argument("out")
    pr.add_argument("--layout", choices=["lrs3", "csv"], default="lrs3")
    pr.add_argument("--limit", type=int, default=0)
    pr.set_defaults(fn=cmd_prepare)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(asctime)s %(name)s %(message)s")
    a.fn(a)


if __name__ == "__main__":
    main()
