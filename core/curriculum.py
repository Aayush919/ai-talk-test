"""Seeded speaking levels — intro first, then why English, routine, family, work."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVELS_PATH = ROOT / "data" / "levels.json"


@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    about: str
    teach: str
    slots: tuple[str, ...]
    questions: tuple[str, ...]
    follow_ups: tuple[str, ...]


@lru_cache(maxsize=1)
def load_lessons() -> tuple[Lesson, ...]:
    raw = json.loads(LEVELS_PATH.read_text(encoding="utf-8"))
    return tuple(
        Lesson(
            id=item["id"],
            title=item["title"],
            about=(item.get("about") or item["title"]).strip(),
            teach=item["teach"],
            slots=tuple(item.get("slots") or ()),
            questions=tuple(item.get("questions") or ()),
            follow_ups=tuple(item.get("follow_ups") or ()),
        )
        for item in raw
    )


def lesson_index(lesson_id: str) -> int:
    ids = [l.id for l in load_lessons()]
    try:
        return ids.index(lesson_id)
    except ValueError:
        return 0


def get_lesson(lesson_id: str) -> Lesson:
    lessons = load_lessons()
    for item in lessons:
        if item.id == lesson_id:
            return item
    return lessons[0]


def next_lesson_id(lesson_id: str) -> str | None:
    lessons = load_lessons()
    i = lesson_index(lesson_id)
    if i + 1 < len(lessons):
        return lessons[i + 1].id
    return None
