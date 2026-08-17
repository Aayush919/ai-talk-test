"""Short-term entity tracking from the current utterance. No Qdrant."""

from __future__ import annotations

import re
from typing import Any

_STOP = frozenset(
    {
        "i",
        "me",
        "my",
        "we",
        "you",
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "that",
        "this",
        "it",
        "was",
        "were",
        "is",
        "am",
        "are",
        "have",
        "had",
        "did",
        "do",
        "just",
        "very",
        "really",
    }
)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'+-]{1,}")


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_entities(text: str, *, previous: list[str] | None = None, limit: int = 8) -> list[str]:
    found: list[str] = []
    for token in _TOKEN.findall(_trim(text)):
        if token.lower() in _STOP:
            continue
        if token[0].isupper() or len(token) >= 5:
            item = token.strip("'")
            if item and item.lower() not in {row.lower() for row in found}:
                found.append(item)
    merged: list[str] = []
    for item in list(previous or []) + found:
        if item.lower() not in {row.lower() for row in merged}:
            merged.append(item)
    return merged[-max(1, limit) :] if merged else []
