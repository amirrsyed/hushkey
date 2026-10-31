# Architecture

```
┌──────────────────────────── Hushkey.app (Swift, macOS 14+) ────────────────────────────┐
│ CameraService  AVCapture 720p@30, un-mirrored, 420f                                     │
│      │                                                                                  │
│ FrameProcessor Vision VNDetectFaceLandmarksRequest (rev 3, 76 pts) → 11 keypoints       │
│      │         face+neck crop → grayscale JPEG (≤320 px wide) + keypoints in crop coords│
│      │                                                                                  │
│ CapturePipeline  0.4 s pre-roll · MotionGate (hands-free) · streams while recording     │
│      │                                                                                  │
│ HotkeyMonitor  hold Right-⌥ = talk · double-tap = hands-free · Esc = cancel             │
│ HUD (NSPanel, non-activating) · TextInserter (⌘V w/ clipboard restore, or keystrokes)   │
└──────┬──────────────────────────────────────────────────────────────────────────────────┘
       │ WebSocket ws://127.0.0.1:8765/v1/stream (JSON)
┌──────▼──────────────────────── hushkey serve (Python) ──────────────────────────────────┐
│ preprocess()  resample → 25 fps · interpolate missed faces                              │
│   ├─ mouth.py   similarity warp to Auto-AVSR mean face → 96×96 → 88×88, μ .421 σ .165   │
│   └─ keypoints.py + throat.py → 16-d kinematic stream                                   │
│ HushNet       Auto-AVSR conformer (frozen) + LoRA + gated kinematic fusion              │
│ Decoder       joint CTC/attention beam (β=10) + LRS3 subword RNN-LM; CTC-greedy partials│
│ cleanup.py    optional local LLM (Ollama) with N-best + personal vocabulary + app       │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Why a separate Python backend?

The only openly downloadable sentence-level lip-reading models that are good enough for dictation
(Auto-AVSR: 19.1 % WER on LRS3; AV-HuBERT: similar but non-commercial) are PyTorch models with ESPnet
or fairseq decoders. Keeping them in Python lets the same code train and serve. The app is a thin,
native client: camera, Vision, hotkeys, HUD and text insertion. The contract between them is
the wire protocol below, so the backend can later be swapped for a Core ML build without changing the app.

## Keypoint schema (shared by Vision and MediaPipe)

| name | Apple Vision (app) | MediaPipe Face Landmarker (training prep) |
|---|---|---|
| `right_eye`, `left_eye` | centroid of `rightEye` / `leftEye` | mean of 33,133,159,145 / 362,263,386,374 |
| `nose_tip` | centroid of `nose` | mean of 1,2,98,327 |
| `mouth_center` | centroid of `outerLips` | mean of 0,17,61,291 |
| `lip_upper/lower_inner` | `innerLips` extremes near the midline (face-aligned) | 13 / 14 |
| `lip_upper/lower_outer` | `outerLips` extremes near the midline | 0 / 17 |
| `lip_left`, `lip_right` | `outerLips` extremes along the eye axis | 291 / 61 |
| `chin` | lowest `faceContour` point (face-aligned) | 152 |

The backend re-sorts eye and mouth-corner pairs by image x, so a mirrored frame can't swap them.
The first four points drive Auto-AVSR's similarity warp, which is exactly what Chaplin derives from
MediaPipe's face detector. That means no MediaPipe or RetinaFace is needed at runtime.

## Kinematic stream (16 channels @ 25 fps)

Face-aligned (eye line horizontal), in inter-ocular-distance units, z-normalised per utterance:

| # | channel | meaning |
|---|---|---|
| 0–5 | `aperture_inner`, `aperture_outer`, `width`, `jaw`, `mouth_drop`, `roundness` | lip/jaw geometry |
| 6–11 | first differences of 0–5 | articulation velocity |
| 12 | `throat_energy` | mean abs frame difference in a patch 0.65 IOD below the chin |
| 13 | `throat_flow_y` | mean vertical Farnebäck flow in that patch (larynx/hyoid up/down) |
| 14 | `head_motion` | nose displacement per frame (lets the model discount head bobbing) |
| 15 | `throat_valid` | 1 if the neck patch is inside the frame |

Per-utterance normalisation removes per-speaker rest shape and landmark-source bias. The stream joins the
network at the 512-d ResNet output through `fusion.enc` (a 3-layer temporal CNN). It is scaled by
`tanh(gate)`, and the gate is initialised to 0, so the model starts identical to Auto-AVSR. `hushkey eval
--compare-zero-kinematics` reports whether the stream helps, using a paired-bootstrap p-value.

## Wire protocol

`GET /v1/health` → `{"ok": true, "device": "cpu", "adapter": ".../personal.pt", "lm": true, ...}`

`WS /v1/stream`, one JSON object per text message:

```jsonc
// client → server
{"type": "start", "app": "com.apple.mail", "partials": true, "enroll_text": null}
{"type": "frame", "t": 1234.567, "jpeg": "<base64 grayscale JPEG>",
 "kp": {"right_eye": [x, y], "left_eye": [x, y], ... 11 names ...} }   // kp: null if no face
{"type": "stop"}      // or {"type": "cancel"}

// server → client
{"type": "ready", "info": {...}}
{"type": "partial", "text": "SEE YOU AT"}                    // ≈1/s while recording
{"type": "final", "text": "See you at five.", "raw": "SEE YOU AT FIVE",
 "alternatives": ["SEE YOU AT NINE", ...], "latency_ms": 840, "vsr_ms": 610,
 "preprocess_ms": 120, "frames": 62, "face_ratio": 1.0, "llm": false,
 "enrolled": "20261031-120102-ab12cd.npz"}                   // only if enroll_text was set
{"type": "error", "message": "no face found in any frame — check lighting and framing"}
```

`POST /v1/transcribe` with `{"frames": [...frame messages...], "app": ..., "enroll_text": ...}` returns
the same `final` / `error` object (handy for scripts and tests).

## Latency budget (push-to-talk)

The app streams frames while you mouth, so on release the server has everything. It then runs preprocessing
(≈2 ms per frame), one encoder pass and beam search. On CPU, beam search over a 250 M-parameter model is the
dominant cost and grows with utterance length. Lower `--beam` (default 10; Chaplin uses 40) or `--no-lm`
for speed. A streaming, chunked encoder and Core ML export are on the roadmap.
