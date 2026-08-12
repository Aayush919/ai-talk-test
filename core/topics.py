"""Predefined practice topics — loaded from data/topics.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOPICS_PATH = ROOT / "data" / "topics.json"


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    prompt: str
    starter: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "prompt": self.prompt,
            "starter": self.starter,
        }


@lru_cache(maxsize=1)
def load_topics() -> tuple[Topic, ...]:
    raw = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    return tuple(
        Topic(
            id=item["id"],
            title=item["title"],
            prompt=item["prompt"],
            starter=item["starter"],
        )
        for item in raw
    )


def get_topic(topic_id: str) -> Topic:
    matched = {t.id: t for t in load_topics()}
    topic = matched.get(topic_id)
    if topic is None:
        known = ", ".join(sorted(matched))
        raise KeyError(f"Unknown topic={topic_id!r}. Known: {known}")
    return topic


def list_topics() -> list[dict[str, str]]:
    return [t.as_dict() for t in load_topics()]
