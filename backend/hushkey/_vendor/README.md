# Vendored code

`espnet/` is the trimmed ESPnet subset shipped with
[Chaplin](https://github.com/amanvirparhar/chaplin) / [Auto-AVSR](https://github.com/mpc001/auto_avsr)
(Apache-2.0, © Shigeki Karita, Johns Hopkins University, Imperial College London and ESPnet contributors).
It defines the conformer VSR model the pretrained `LRS3_V_WER19.1` checkpoint expects, plus
joint CTC/attention beam search and the RNN language model.

Hushkey changes (all marked `hushkey:` where non-trivial):
- imports rewritten from `espnet.` to `hushkey._vendor.espnet.`
- removed `distutils` (gone in Python 3.12) from `ctc.py`, `mask.py`, `cli_utils.py`, `e2e_asr_transformer.py`
- `lm/default.py` imports `to_device` from `nets_utils` (the original `e2e_asr` module isn't vendored)
- dropped the plotting/AV modules that the visual-only model doesn't use

`../assets/unigram5000*` (SentencePiece model + units) come from Auto-AVSR (Apache-2.0);
`../assets/20words_mean_face.npy` (mean-face reference for mouth alignment) from Auto-AVSR/Chaplin.
