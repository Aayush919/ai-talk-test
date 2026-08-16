"""In-memory session — transcript in Mongo, no local audio files."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from core.memory.bank import MemoryBank
from core.topics import Topic
from wrappers.mongo_store import MongoStore

Message = dict[str, str]


@dataclass
class Session:
    session_id: str
    mode: str
    messages: list[Message] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    topic: Topic | None = None
    turn: int = 0
    learner_id: str = ""
    memory: MemoryBank | None = None

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def snapshot(self) -> list[Message]:
        return list(self.messages)


def new_session(
    mode: str,
    mongo: MongoStore,
    topic: Topic | None = None,
    learner_id: str = "",
) -> Session:
    session_id = uuid.uuid4().hex[:12]
    lid = (learner_id or "").strip() or session_id
    mongo.create_session(
        session_id,
        mode,
        topic_id=topic.id if topic else None,
        topic_title=topic.title if topic else None,
    )
    memory = MemoryBank(learner_id=lid, topic_id=topic.id if topic else "introduction")
    return Session(
        session_id=session_id,
        mode=mode,
        topic=topic,
        learner_id=lid,
        memory=memory,
    )
