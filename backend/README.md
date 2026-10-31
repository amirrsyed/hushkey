# hushkey (backend)

Local model server, training and evaluation for [Hushkey](../README.md), a silent-speech webcam dictation app for macOS.

```bash
pip install -e ".[dev]"     # + ".[prepare]" for offline video preparation (MediaPipe)
hushkey download            # pretrained Auto-AVSR VSR model + LM
hushkey serve               # ws://127.0.0.1:8765/v1/stream for the app
pytest -q                   # synthetic-face tests, no weights needed
```

See `../docs/ARCHITECTURE.md` and `../docs/TRAINING.md`.
