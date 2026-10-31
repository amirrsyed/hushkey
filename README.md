# Hushkey
![Hushkey Logo](hushkey.png)
[![CI](https://github.com/amirrsyed/hushkey/actions/workflows/ci.yml/badge.svg)](https://github.com/amirrsyed/hushkey/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Release](https://img.shields.io/github/v/release/amirrsyed/hushkey)](https://github.com/amirrsyed/hushkey/releases/latest)

**Silent dictation for macOS.** Hold a key, mouth your words with no sound, release, and the text
appears at your cursor in any app. It works like Wispr Flow, except it reads your lips, jaw and throat
from the webcam instead of listening.

> **Status: proof of concept.** The pipeline runs end to end: native Mac app → local model server →
> text at the cursor, with adapter training on your own clips. Accuracy depends on the pretrained lip
> reader and on how much you personalise it; see [Honest expectations](#honest-expectations).
> Everything runs locally. No audio is used and no video leaves your Mac.

```
 hold Right-⌥ ──► webcam 30 fps ──► Apple Vision face landmarks ──► mouth + neck crop ──┐
                                                                                        │ localhost
       text typed at cursor ◄── optional local LLM ◄── beam search + LM ◄── HushNet ◄───┘
                                                                         (Auto-AVSR VSR
                                                                          + LoRA adapters
                                                                          + lip/jaw/throat
                                                                            kinematic branch)
```

## What's new here

| | |
|---|---|
| **A native Mac app** | Menu-bar app with webcam permission, push-to-talk on a modifier key, double-tap for hands-free, Esc to cancel, a non-activating HUD with a live mouth preview and partial transcript, and paste-at-cursor with clipboard restore. |
| **HushNet** | The Auto-AVSR conformer lip reader, **frozen**, plus LoRA adapters and a new **kinematic branch**: lip aperture/width, jaw drop and their velocities, plus **throat motion** (energy and vertical optical flow below the chin). It enters through a zero-initialised gate, so the model starts *exactly* as the pretrained one and only uses the extra stream if it lowers the loss. |
| **Silent-speech adaptation** | Voiced corpora (LRS3, VoxCeleb2) are pushed toward silent webcam speech with hypo-articulation, tempo and webcam augmentations. |
| **Personalisation in the app** | "Train on my mouth" records ~40 silent prompts through the live pipeline and trains a few-MB personal adapter on your Mac. |
| **RL fine-tuning** | Minimum-word-error-rate training, i.e. sequence-level policy gradient on beam-search N-best with reward = −word errors. |
| **Measurable claims** | `hushkey eval --compare-…` gives a paired-bootstrap p-value: does the adapter help, does the throat stream help? |

## Built on (and what was taken from each)

| repo | used for | not used, and why |
|---|---|---|
| [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) | **Backbone model and training recipe**: conformer VSR, joint CTC/attention loss, mouth-ROI preprocessing, LRS3 SentencePiece units, frame-budget bucketing, adaptive time masking | Lightning trainer (replaced by an adapter-only loop) |
| [amanvirparhar/chaplin](https://github.com/amanvirparhar/chaplin) | Proved the real-time loop; Hugging Face mirror of the `LRS3_V_WER19.1` weights + RNN-LM; vendored ESPnet decoder; the 4-point (eyes, nose, mouth) alignment; LLM-cleanup idea | OpenCV window, `pynput`, MediaPipe detector: replaced by a native app with Apple Vision. Cleanup now also gets N-best, personal vocabulary and the target app |
| [joeblau/subvox](https://github.com/joeblau/subvox) | Mac-app structure for silent-speech prototypes; **energy-gate segmentation** with calibrated noise floor, pre-roll and hang-over (`MotionGate.swift`); the *"is the signal real?"* statistical test | AirPods head-motion input and DTW templates (small-vocabulary); listed as a roadmap channel |
| [facebookresearch/av_hubert](https://github.com/facebookresearch/av_hubert) | Considered as the backbone (similar VSR accuracy) | **Non-commercial license** and a fairseq fork that's painful on Apple Silicon. The adapter design would work on it too |
| [LipNet-PyTorch](https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch) | Reference for CTC sentence lip reading and Conv3D front-ends | Trained on GRID's fixed 6-word grammar, so it can't do open dictation |
| [hilab-open-source/WatchYourMouth](https://github.com/hilab-open-source/WatchYourMouth) | Evidence that silent speech is decodable from geometry rather than RGB (depth point clouds). It motivated the landmark-geometry kinematic branch and per-utterance normalisation | Needs a TrueDepth/depth sensor and CUDA PointNet++ ops; a depth channel is on the roadmap |

## Install

Requirements: macOS 14+ (Apple Silicon recommended). Hushkey uses Python 3.10–3.13 if you have it;
otherwise it downloads a private copy for itself. About 2.5 GB of disk space.

### Option A: download the app (no Terminal)

1. Download **Hushkey.dmg** from the [Releases](../../releases) page and drag Hushkey into Applications.
2. The first time, **right-click Hushkey ▸ Open ▸ Open**. The build isn't notarized by Apple, so a plain
   double-click is blocked once. If macOS says it's "damaged", run `xattr -dr com.apple.quarantine /Applications/Hushkey.app`.
3. In the welcome window, allow **Camera** and **Accessibility**, then click **Install model server**
   (one-time, ≈1.5 GB, a few minutes).

### Option B: from source (one command)

```bash
git clone https://github.com/amirrsyed/hushkey && cd hushkey
xcode-select --install     # Apple's build tools, if you don't have them
./install.sh               # model server + app in /Applications, then opens it
```

Re-run `./install.sh` to update and `./uninstall.sh` to remove. After installing, the source folder is no
longer needed: everything lives in `/Applications/Hushkey.app` and `~/Library/Application Support/Hushkey`.
`make` lists the shortcuts (`make install`, `make dmg`, `make test`, …).

### After installing

- **Open at login**: Hushkey ▸ Settings ▸ General.
- **If text shows in the overlay but isn't typed**: macOS isn't letting Hushkey type. Open System Settings ▸
  Privacy & Security ▸ Accessibility, remove Hushkey with **–**, add it again, and relaunch. Each new
  unsigned build counts as a new app. Until then Hushkey copies the text to your clipboard and tells you to press ⌘V.
- **Optional LLM cleanup** (as in Chaplin): install [Ollama](https://ollama.com), `ollama pull qwen3:4b`,
  and append `--llm` to the command in Settings ▸ Backend.

### For developers

`make dev` makes an editable backend in `backend/.venv` and points the app at it. `make app` builds the app
only. `hushkey serve --tiny` runs a random tiny model for UI work without the 1 GB download (the output is
gibberish). CI builds the app and runs the tests. Pushing a tag such as `v0.2.0` makes the Release workflow
publish `Hushkey.dmg`.

### Publishing your own copy on GitHub

```bash
# create an EMPTY repo named hushkey at https://github.com/new (no README/license), then:
cd hushkey
git remote add origin https://github.com/amirrsyed/hushkey.git
git push -u origin main
git tag v0.1.0 && git push origin v0.1.0     # optional: builds Hushkey.dmg into a GitHub Release
```

## Using it

| action | how |
|---|---|
| Dictate | hold **Right ⌥** (configurable: Right ⌘, fn), mouth silently, release |
| Hands-free | **double-tap** the key; tap once more to finish |
| Cancel | **Esc** while recording |
| Auto-detect | menu ▸ *Auto-detect mouthing*: starts/stops on lip and jaw motion (camera stays on) |
| Personalise | menu ▸ *Train on my mouth…* → record ~40 prompts → *Train personal adapter* |

Tips: face the camera with your **mouth and neck in view** and even lighting. Exaggerate slightly, and
dictate phrases rather than single words (more context helps the decoder). The camera turns off 30 s after
your last dictation unless auto-detect is on.

## Training your own model

See [docs/TRAINING.md](docs/TRAINING.md). In short:

```bash
hushkey prepare /data/lrs3/trainval data/lrs3 --layout lrs3      # same preprocessing as live
hushkey train adapt --manifest data/lrs3/manifest.jsonl          # → adapters/silent.pt   (GPU)
hushkey train personalize                                        # → adapters/personal.pt (your enrollment clips)
hushkey train mwer                                               # RL fine-tune on word error rate
hushkey eval <manifest> --adapter …/personal.pt --compare-adapter none
```

## Repository layout

```
app/                       Swift menu-bar app (SwiftPM package + xcodegen project.yml)
  Sources/Hushkey/
    App/                   HushkeyApp (scenes), AppModel (state machine)
    Capture/               CameraService, CapturePipeline (pre-roll, streaming), MotionGate
    Vision/                FaceKeypoints (Vision → 11 keypoints), FrameProcessor (crop + JPEG)
    Backend/               BackendClient (WebSocket), BackendProcess (launch/train)
    Input/                 HotkeyMonitor, TextInserter
    UI/                    Menu, HUD, Settings, Onboarding, Enrollment
backend/
  hushkey/
    preprocess/            keypoints, mouth ROI (Auto-AVSR), throat motion, silent-speech augment
    model/                 backbone loader, LoRA, HushNet, decoding, MWER, tokenizer
    data/                  datasets + bucketing, MediaPipe prep for offline video
    train.py evaluate.py   three training stages, WER + paired bootstrap
    server.py engine.py    FastAPI WebSocket server, inference runtime
    cleanup.py             casing/punctuation + optional Ollama LLM
    _vendor/espnet         Apache-2.0 decoder/model code from Auto-AVSR/Chaplin (patched for py3.12+)
  tests/                   synthetic talking faces → preprocessing, model, training, server
docs/                      ARCHITECTURE.md (protocol, schemas), TRAINING.md
scripts/                   install_backend.sh (also bundled in the app), build_app.sh, package_dmg.sh, setup_backend.sh (dev)
install.sh uninstall.sh    one-command install / removal; Makefile wraps everything
.github/workflows/         ci.yml (tests, app build, installer smoke test), release.yml (tag → Hushkey.dmg)
```

## Honest expectations

- The base lip reader reports **19.1 % WER on LRS3** (voiced TED talks; Auto-AVSR's figure). Silent
  mouthing into a laptop webcam is harder: articulation is smaller and lighting worse. Expect clearly
  worse results until you've run `personalize`, and measure with `hushkey eval`.
- Lip reading can't separate visually identical sounds (/p b m/, /f v/, /t d n/). The LM and the
  optional LLM cleanup resolve some of this from context.
- Throat motion seen by an RGB webcam is a weak cue. It's gated, ablatable and measured, not assumed to help.
- Not yet: streaming partials from a chunked encoder (current partials re-decode the buffer once a
  second), Core ML/ANE export, punctuation commands, multiple languages.

## Roadmap

- Developer ID signing + notarization in the release workflow: no right-click-to-open, and permissions survive updates
- Core ML export of the frontend and encoder (run on the Neural Engine, drop the Python dependency for inference)
- Chunked streaming encoder for word-by-word output
- Extra channels behind the same gate: AirPods head/jaw motion (Subvox) and TrueDepth depth (Watch Your Mouth)
- Voice-free commands ("new line", "delete that") via a small intent head on the CTC posteriors

## Privacy

Frames are processed in memory by the local server and discarded after each utterance. The only exception
is the enrollment clips you choose to record, which are saved as mouth crops plus keypoint features
(not full frames) under `~/Library/Application Support/Hushkey/enrollment/`. Delete that folder to remove
them. The optional LLM runs through your local Ollama.

## License

Hushkey code: MIT. Vendored ESPnet/Auto-AVSR code: Apache-2.0. Pretrained weights are **downloaded, not
redistributed**, and carry the terms of their training data (research use). See [NOTICE](NOTICE).

## Citations

- Ma et al., *Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels*, ICASSP 2023.
- Petridis et al., *Visual-Only Recognition of Normal, Whispered and Silent Speech*, ICASSP 2018.
- Prabhavalkar et al., *Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models*, ICASSP 2018.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021.
- Wang et al., *Watch Your Mouth: Silent Speech Recognition with Depth Sensing*, CHI 2024.
- Shi et al., *Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction* (AV-HuBERT), ICLR 2022.
