"""Normalize LLM / UI text for Windows consoles + TTS."""

from __future__ import annotations

import re

_STAGE = re.compile(
    r"\([^)]*(?:smil|laugh|grin|pause|sigh|chuckl)[^)]*\)",
    re.I,
)
_RECAP_ASK = re.compile(
    r"\b(do you know|tell me about me|what do you know|say it together|"
    r"put it (?:all )?together|practice saying)\b",
    re.I,
)

_TRANSLATE = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen (charmap crash on Windows)
        "\u2012": "-",
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2212": "-",
        "\u00a0": " ",
        "\u202f": " ",
        "\u2026": "...",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def clean_speech_text(text: str) -> str:
    """ASCII-friendly text for TTS + Windows logging."""
    if not text:
        return ""
    return " ".join(str(text).translate(_TRANSLATE).split())


def clip_spoken_reply(text: str, *, user_text: str = "") -> str:
    """Short spoken default; allow a bit more only when they ask for recap/practice."""
    raw = _STAGE.sub(" ", clean_speech_text(text))
    raw = " ".join(raw.split())
    if not raw:
        return ""
    allow_long = bool(_RECAP_ASK.search(user_text or ""))
    has_fix = bool(
        re.search(r"['\"][^'\"]{6,}['\"]", raw)
        or re.search(
            r"\b(we don't say|say it like this|try it|you can say|"
            r"please repeat|a more natural way|let's say it this way)\b",
            raw,
            re.I,
        )
    )
    max_words = 44 if (allow_long or has_fix) else 28
    parts = re.split(r"(?<=[.!?])\s+", raw)
    keep = parts[:4] if (allow_long or has_fix) else parts[:2]
    out = " ".join(p.strip() for p in keep if p.strip())
    words = out.split()
    if len(words) <= max_words:
        return out
    cut = words[:max_words]
    joined = " ".join(cut)
    if not joined.endswith((".", "?", "!")):
        joined += "."
    return joined


def safe_print(*args: object, **kwargs: object) -> None:
    """print() that never crashes on Windows charmap."""
    try:
        print(*args, **kwargs)  # type: ignore[arg-type]
    except UnicodeEncodeError:
        flat = " ".join(clean_speech_text(str(a)) for a in args)
        print(flat.encode("ascii", "replace").decode("ascii"), **kwargs)  # type: ignore[arg-type]
