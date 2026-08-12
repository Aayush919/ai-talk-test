"""Talk mode protocol — strategies, not if/else trees."""

from __future__ import annotations

from typing import Protocol

from core.session import Session


class TalkMode(Protocol):
    name: str

    def listen(self, session: Session, turn: int) -> str:
        """Capture user speech → return transcript text."""
        ...
