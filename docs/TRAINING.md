# Training the Hushkey model

Hushkey doesn't train a lip reader from scratch. 3,000+ hours of labelled talking faces (what Auto-AVSR used)
is out of reach for a proof of concept. Instead it starts from the strongest open VSR checkpoint and trains
**small adapters** for the two things that checkpoint was never trained on:

1. **Silent speech** — LRS3, VoxCeleb2 and AVSpeech are *voiced* speech. When people mouth without sound
   they articulate less and more slowly. Petridis et al. (ICASSP 2018) measured up to an 8.5-point
   accuracy drop going from normal to silent speech.
2. **You** — your face, your webcam, your lighting and your vocabulary.

Everything below trains only LoRA weights on the encoder/decoder attention and feed-forward
layers, plus the kinematic fusion branch: about 1–2 % of the 250 M parameters. The base weights never change,
and an adapter is a few MB.

| stage | data | objective | output |
|---|---|---|---|
| `adapt` | public talking-face corpora + silent-speech augmentation | joint CTC/attention CE (Auto-AVSR's loss) | `adapters/silent.pt` |
| `personalize` | your enrollment clips (+ optional replay of public data) | same | `adapters/personal.pt` |
| `mwer` | your clips | minimum word error rate (sequence-level RL on beam N-best) + CE anchor | `adapters/personal.pt` |

The server loads `personal.pt` if it exists, otherwise `silent.pt`, otherwise the plain base model.

## 0. Base weights

```bash
hushkey download          # Auto-AVSR LRS3_V_WER19.1 + LRS3 subword RNN-LM (Chaplin's HF mirror)
```

## 1. `adapt`: silent-speech domain adaptation on public data

### Data

Any corpus of frontal talking faces with sentence transcripts works. The useful ones:

| corpus | size | access | notes |
|---|---|---|---|
| **LRS3** (TED/TEDx) | ~430 h | free for research after signing the agreement with the Oxford VGG group | what the base model was fine-tuned on; the reference |
| **VoxCeleb2** + automatic transcripts | ~1,300 h (English subset) | research | Auto-AVSR ships the Whisper-style auto-label lists |
| **Petridis et al. 2018** normal / whispered / silent database | small (digits + phrases) | research | the only public set with *true silent* mouthing; ideal as a held-out silent test set |
| **Your own recordings** | any | — | CSV of `path,text` |

Hushkey can't redistribute any of these. Download them under their own terms, then convert:

```bash
pip install -e "backend[prepare]"          # MediaPipe Face Landmarker for offline keypoints
hushkey prepare /data/lrs3/trainval  data/lrs3_trainval --layout lrs3
hushkey prepare /data/my_clips.csv   data/mine          --layout csv
```

`prepare` runs the **same** `preprocess()` as the live server (mouth crops, kinematic and throat stream).
It writes `.npz` files plus `manifest.jsonl`, so training and inference see identical inputs.

### Augmentation (what makes this a *silent-speech* model)

`SilentSpeechAugment` (`preprocess/augment.py`), applied on the fly:

- **hypo-articulation**: `v' = m + α(v − m)`, where `m` is a 9-frame moving average and α∈[0.45, 0.9].
  It is applied to pixels and to the geometry channels together, which shrinks articulatory excursions
  the way silent speech does.
- **tempo** 0.8–1.25× (silent speech is slower)
- **webcam** JPEG quality 20–70, gamma, blur, sensor noise
- Auto-AVSR's adaptive time masking and random 88×88 crop
- **kinematic dropout**: throat channels zeroed 30 % of the time, so the model never relies on a neck
  being visible

### Run

```bash
hushkey train adapt --manifest data/lrs3_trainval/manifest.jsonl \
                    --manifest data/vox2/manifest.jsonl \
                    --device cuda --max-frames 1600
```

Defaults: 20k steps, AdamW lr 5e-4, 500 warm-up, cosine decay, frame-budget bucketing like Auto-AVSR. It
reports CTC-greedy WER on a 5 % hold-out every 500 steps and keeps the best adapter. The backbone is
frozen, but gradients still flow through it to reach the LoRA weights. Use a single 24 GB+ CUDA GPU and
lower `--max-frames` if you run out of memory.

## 2. `personalize`: your mouth, your webcam

In the app, open **Hushkey ▸ Train on my mouth…** and mouth the prompts silently while holding the
button or the hotkey. Each clip goes through the live pipeline and is saved with its prompt as a training
example in `~/Library/Application Support/Hushkey/enrollment/`. After about 40 clips, click **Train
personal adapter**, or run:

```bash
hushkey train personalize                                  # uses the enrollment manifest
hushkey train personalize --replay data/lrs3_trainval/manifest.jsonl   # mix 30 % public data to avoid forgetting
PYTORCH_ENABLE_MPS_FALLBACK=1 hushkey train personalize --device mps   # Apple-Silicon GPU (CTC falls back to CPU)
```

Defaults: 600 steps, lr 3e-4, a 15 % hold-out with WER logged before (step 0) and during training, and
milder augmentation, since your clips are already silent webcam speech. It starts from `silent.pt` if present.

## 3. `mwer`: optimise the metric you actually feel

Cross-entropy trains one token at a time; dictation quality is judged in whole words. `mwer`
(`model/mwer.py`) does sequence-level RL:

1. Beam search (no gradient) gives an N-best list `y₁…y_N` for each clip.
2. The model re-scores each hypothesis with gradient: `log p(y|x)`, decoder plus CTC.
3. It renormalises over the list and minimises the expected word errors relative to the list mean:
   `L = Σᵢ p̂(yᵢ|x)·(W(yᵢ,y*) − W̄) + λ·L_CE` (Prabhavalkar et al., 2018; λ = 0.1).

```bash
hushkey train mwer --steps 300
```

Hold-out WER uses beam search, so it matches what the app does.

## 4. Did it help? Evaluate with a significance test

```bash
# personal adapter vs the plain base model, same clips, beam + LM
hushkey eval ~/Library/Application\ Support/Hushkey/enrollment/manifest.jsonl \
    --adapter ~/Library/Application\ Support/Hushkey/adapters/personal.pt --compare-adapter none

# is the throat/kinematic stream doing anything?
hushkey eval data/silent_test/manifest.jsonl --compare-zero-kinematics
```

Each run prints both WERs and `p_value_main_not_better` from a paired bootstrap over utterances. This is
Subvox's "is the signal real?" idea applied to sentence WER. Don't trust a gain with p > 0.05.
Evaluate on clips you didn't train on.

## Honest expectations

- The base model reports 19.1 % WER on LRS3 test (voiced, broadcast-quality, Auto-AVSR's numbers).
  Silent mouthing into a laptop webcam will start noticeably worse. That gap is what the adapters target.
- Lip reading is inherently ambiguous (/p b m/ look identical). The LM, the optional LLM cleanup with
  your vocabulary, and short dictation-style phrases help most.
- Throat motion from an RGB webcam is a weak signal (no vocal-fold vibration in silent speech). It's
  included because it's cheap, gated and measurable, not because it's expected to dominate.
