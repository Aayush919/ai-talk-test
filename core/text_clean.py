"""Normalize LLM / UI text for Windows consoles + TTS."""

from __future__ import annotations


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


def safe_print(*args: object, **kwargs: object) -> None:
    """print() that never crashes on Windows charmap."""
    try:
        print(*args, **kwargs)  # type: ignore[arg-type]
    except UnicodeEncodeError:
        flat = " ".join(clean_speech_text(str(a)) for a in args)
        print(flat.encode("ascii", "replace").decode("ascii"), **kwargs)  # type: ignore[arg-type]
