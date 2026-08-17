"""In-memory session — live-call transcript only, nothing persisted."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

Message = dict[str, str]


@dataclass
class Session:
    session_id: str
    mode: str
    messages: list[Message] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    turn: int = 0
    learner_id: str = ""
    current_topic: dict | None = None
    conversation_id: str | None = None

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def snapshot(self) -> list[Message]:
        return list(self.messages)


def new_session(
    mode: str,
    learner_id: str = "",
) -> Session:
    session_id = uuid.uuid4().hex[:12]
    return Session(
        session_id=session_id,
        mode=mode,
        learner_id=(learner_id or "").strip(),
    )
