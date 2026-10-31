"""Turn raw lip-reading output into dictation-ready text.

Raw VSR output is upper-case with no punctuation and a fair number of homophene errors
("BAT"/"PAT"/"MAT" look identical on the lips). Like Chaplin, we optionally ask a small
local LLM (via Ollama's HTTP API — nothing leaves the machine) to fix obvious
mis-readings and punctuate. Unlike Chaplin we pass the N-best list, the user's personal
vocabulary and the frontmost app, which give the LLM real evidence to choose with.
Without Ollama we fall back to deterministic casing + punctuation.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

SYSTEM_PROMPT = """You correct the output of a silent lip-reading dictation system.
The user mouthed words without sound; a visual speech model transcribed them. Lip-reading confuses
words that look alike on the lips (p/b/m, f/v, t/d/n, k/g, ch/j/sh), so the transcript and its
alternatives may contain such substitutions.

Rules:
- Return what the user most plausibly mouthed. Prefer the top transcript; use the alternatives and
  the personal vocabulary as evidence when a word looks wrong.
- Only replace words that look mis-read; do not add content, do not rephrase, do not answer it.
- Use normal capitalisation and punctuation suitable for the target app.
Respond with JSON: {"text": "..."}"""


@dataclass
class CleanupConfig:
    enabled: bool = False
    model: str = "qwen3:4b"
    host: str = "http://127.0.0.1:11434"
    timeout: float = 6.0
    vocabulary: list[str] = field(default_factory=list)


def basic_format(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    s = text.lower()
    s = re.sub(r"\bi\b", "I", s)
    s = re.sub(r"\bi'(m|ll|d|ve)\b", lambda m: "I'" + m.group(1), s)
    s = s[0].upper() + s[1:]
    if s[-1] not in ".?!":
        first = s.split(" ", 1)[0].lower()
        s += "?" if first in {"what", "why", "how", "when", "where", "who", "which", "is", "are", "can", "could", "would", "should", "do", "does", "did", "will"} else "."
    return s


def llm_cleanup(raw: str, alternatives: list[str], cfg: CleanupConfig, app: Optional[str] = None) -> Optional[str]:
    if not cfg.enabled or not raw.strip():
        return None
    user = {"transcript": raw, "alternatives": alternatives[:4], "personal_vocabulary": cfg.vocabulary[:200], "target_app": app or "unknown"}
    body = {
        "model": cfg.model,
        "stream": False,
        "format": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "think": False,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(user)}],
        "options": {"temperature": 0.1},
    }
    req = urllib.request.Request(cfg.host.rstrip("/") + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout) as r:
            msg = json.loads(r.read())["message"]["content"]
        text = json.loads(msg).get("text", "").strip()
        return text or None
    except Exception:
        return None


def finalize(raw: str, alternatives: list[str], cfg: CleanupConfig, app: Optional[str] = None) -> tuple[str, bool]:
    """Returns (text, used_llm)."""
    cleaned = llm_cleanup(raw, alternatives, cfg, app)
    if cleaned:
        return cleaned, True
    return basic_format(raw), False
