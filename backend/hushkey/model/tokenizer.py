"""Text ↔ token ids with Auto-AVSR's LRS3 SentencePiece unigram-5000 model."""

from __future__ import annotations

import re

import torch

from ..config import SPM_MODEL_FILE, TOKENS_FILE


def normalize_text(text: str) -> str:
    """LRS3 transcripts are upper-case words with apostrophes and no other punctuation."""
    text = text.upper().replace("’", "'")
    text = re.sub(r"[^A-Z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Tokenizer:
    def __init__(self):
        import sentencepiece

        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(SPM_MODEL_FILE))
        units = TOKENS_FILE.read_text(encoding="utf8").splitlines()
        self.hashmap = {u.split()[0]: int(u.split()[-1]) for u in units}

    def encode(self, text: str) -> torch.Tensor:
        pieces = self.spm.EncodeAsPieces(normalize_text(text))
        return torch.tensor([self.hashmap.get(p, self.hashmap["<unk>"]) for p in pieces], dtype=torch.long)
