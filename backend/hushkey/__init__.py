"""Hushkey: silent-speech (no audio) dictation from a webcam.

The backend turns a stream of webcam frames + face keypoints into text:

    frames + keypoints ──► mouth crops (Auto-AVSR preprocessing)
                      └──► kinematic stream (lip/jaw geometry + throat motion)
                                   │
                 HushNet = Auto-AVSR VSR conformer (frozen)
                           + LoRA adapters (silent-speech / personal)
                           + gated kinematic fusion
                                   │
                 joint CTC/attention beam search (+ optional RNN LM)
                                   │
                 optional LLM cleanup ──► text typed at the cursor
"""

__version__ = "0.1.0"
