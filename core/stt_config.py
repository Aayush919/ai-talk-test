"""Deepgram STT options from env — nova-3 + language chain + endpointing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepgramSTTConfig:
    model: str
    language: str
    language_fallbacks: tuple[str, ...]
    endpointing_ms: int

    @property
    def language_chain(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for code in (self.language, *self.language_fallbacks):
            if code and code not in seen:
                seen.add(code)
                ordered.append(code)
        return tuple(ordered)
