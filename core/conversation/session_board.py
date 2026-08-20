"""In-call topic board in RAM (LangGraph checkpoint). No Mongo, no Redis."""

from __future__ import annotations

import re
from typing import Any

from core.conversation.engagement import (
    detect_user_intent,
    is_confused_turn,
    is_low_content_turn,
    is_memory_probe,
)

CHAPTERS = (
    ("morning", ("wake_up", "morning")),
    ("day", ("work_or_study_day",)),
    ("evening", ("evening",)),
    ("night", ("sleep",)),
)

_HINTS = (
    (
        "wake_up",
        re.compile(
            r"\b(?:wake|woke|get up)\b|\b\d{1,2}\s*[:.]\s*\d{2}\b|"
            r"\b\d{1,2}\s*(?:am|pm|o'?clock)\b",
            re.I,
        ),
    ),
    (
        "morning",
        re.compile(
            r"\bmorning\b|\byoga\b|\bshower\b|\bbreakfast\b|\bbrush\b|"
            r"\btea\b|\bcoffee\b|\bexercise\b|\basana",
            re.I,
        ),
    ),
    (
        "work_or_study_day",
        re.compile(
            r"\boffice\b|\b(?:work|working)(?!\s+out)\b|\bcoding\b|"
            r"\bdeveloper\b|\bstudy\b|\bcollege\b|\bjob\b",
            re.I,
        ),
    ),
    (
        "evening",
        re.compile(
            r"\bevening\b|\bdinner\b|\bwatch(?:ing)? (?:tv|television|netflix)\b|"
            r"\bafter work\b|\bafter office\b|\bgo for a walk\b",
            re.I,
        ),
    ),
    (
        "sleep",
        re.compile(
            r"\bsleep\b|\bgo to bed\b|\bgo to sleep\b|\bsleep at\b|\bgood night\b",
            re.I,
        ),
    ),
)

_CHAPTER_OF = {
    key: name for name, members in CHAPTERS for key in members
}
_CHAPTER_PRIORITY = ("evening", "morning", "day", "night")


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _goal_keys(topic_goals: list[Any]) -> list[str]:
    keys: list[str] = []
    for goal in topic_goals or []:
        if isinstance(goal, dict) and goal.get("key"):
            keys.append(str(goal["key"]))
        elif isinstance(goal, str) and goal:
            keys.append(goal)
    return keys


def empty_call_board() -> dict[str, Any]:
    return {"chapter": None, "answered": {}, "known": []}


def uses_routine_chapters(topic_goals: list[Any]) -> bool:
    allowed = set(_goal_keys(topic_goals))
    return "wake_up" in allowed or "morning" in allowed


def _remember_line(text: str, known: list[str]) -> list[str]:
    line = _trim(text)
    if (
        not line
        or is_low_content_turn(line)
        or is_memory_probe(line)
        or is_confused_turn(line)
    ):
        return known
    if line.lower() in {item.lower() for item in known}:
        return known
    return (known + [line])[-12:]


def _first_incomplete_chapter(answered: dict[str, Any], allowed: set[str]) -> str | None:
    if "wake_up" not in allowed and "morning" not in allowed:
        return None
    for name, members in CHAPTERS:
        present = [key for key in members if key in allowed]
        if not present:
            continue
        if any(key not in answered for key in present):
            return name
    return None


def detect_spoken_chapter(text: str, allowed: set[str]) -> str | None:
    hits: set[str] = set()
    raw = _trim(text)
    if not raw:
        return None
    for key, pattern in _HINTS:
        if key in allowed and pattern.search(raw):
            chapter = _CHAPTER_OF.get(key)
            if chapter:
                hits.add(chapter)
    for name in _CHAPTER_PRIORITY:
        if name in hits:
            return name
    return None


def update_call_board(
    *,
    board: dict[str, Any] | None,
    user_text: str,
    topic_goals: list[Any],
    current_goal_id: str | None,
    last_question: str | None = None,
    skip_slots: bool = False,
) -> dict[str, Any]:
    """Follow what they just talked about. Do not drag back to an old empty slot."""
    _ = last_question
    data = dict(board or empty_call_board())
    answered = dict(data.get("answered") or {})
    known = [str(item) for item in (data.get("known") or []) if str(item).strip()]
    allowed = set(_goal_keys(topic_goals))
    text = _trim(user_text)
    skip = skip_slots or is_memory_probe(text) or is_confused_turn(text)
    spoken = None
    current = _trim(current_goal_id)
    repeat = detect_user_intent(text) == "REPEAT_COMPLAINT"
    if text and not skip:
        known = _remember_line(text, known)
        if not is_low_content_turn(text) or repeat:
            hinted: set[str] = set()
            for key, pattern in _HINTS:
                if key in allowed and pattern.search(text):
                    hinted.add(key)
                    if key not in answered:
                        answered[key] = text[:80]
            spoken = detect_spoken_chapter(text, allowed)
            other_hint = bool(hinted - ({current} if current else set()))
            if (
                current
                and current in allowed
                and current not in answered
                and not other_hint
            ):
                answered[current] = text[:80]
    data["answered"] = answered
    data["known"] = known
    if spoken:
        data["chapter"] = spoken
    elif not data.get("chapter"):
        data["chapter"] = _first_incomplete_chapter(answered, allowed)
    else:
        open_chapter = _first_incomplete_chapter(answered, allowed)
        members = next((item for name, item in CHAPTERS if name == data.get("chapter")), ())
        present = [key for key in members if key in allowed]
        if open_chapter and present and all(key in answered for key in present):
            data["chapter"] = open_chapter
    return data


def pin_session_goals(
    *,
    topic_goals: list[Any],
    goals_completed: list[Any],
    goals_remaining: list[Any],
    board: dict[str, Any],
) -> dict[str, Any]:
    keys = _goal_keys(topic_goals)
    answered = set((board or {}).get("answered") or {})
    completed = [key for key in keys if key in set(goals_completed) or key in answered]
    remaining_all = [
        key
        for key in (list(goals_remaining) if goals_remaining else keys)
        if key in set(keys) and key not in set(completed)
    ]
    if not remaining_all:
        remaining_all = [key for key in keys if key not in set(completed)]
    chapter = (board or {}).get("chapter")
    members = ()
    if chapter:
        members = tuple(
            key
            for key in next((item for name, item in CHAPTERS if name == chapter), ())
            if key in set(keys)
        )
    incomplete_here = [key for key in members if key not in set(completed)]
    if incomplete_here:
        current = incomplete_here[0]
        remaining = incomplete_here + [
            key for key in remaining_all if key not in set(incomplete_here)
        ]
    else:
        remaining = remaining_all
        current = remaining[0] if remaining else (completed[-1] if completed else None)
    index = keys.index(current) if current in keys else None
    return {
        "goalsCompleted": completed,
        "goalsRemaining": remaining,
        "currentGoalId": current,
        "currentGoalIndex": index,
        "callChapter": chapter,
    }


def allow_goal_switch(board: dict[str, Any] | None, target: str) -> bool:
    data = board or {}
    chapter = data.get("chapter")
    if not chapter:
        return True
    members = next((item for name, item in CHAPTERS if name == chapter), ())
    return target in members


def board_known_lines(board: dict[str, Any] | None) -> list[str]:
    data = board or {}
    lines: list[str] = []
    seen: set[str] = set()
    for key, value in dict(data.get("answered") or {}).items():
        text = _trim(value)
        if not text:
            continue
        label = f"{key.replace('_', ' ')}: {text}"
        lines.append(label)
        seen.add(text.lower())
    for item in data.get("known") or []:
        text = _trim(item)
        if text and text.lower() not in seen:
            lines.append(text)
            seen.add(text.lower())
    return lines[:10]


_GOAL_QUESTIONS = {
    "wake_up": "What time do you usually wake up?",
    "morning": "What do you usually do in the morning after you wake up?",
    "work_or_study_day": "What do you do during the day — work or study?",
    "evening": "What do you usually do in the evening?",
    "sleep": "What time do you usually go to sleep?",
    "name": "What should I call you?",
    "location": "Where are you from?",
    "education_or_work": "Do you work or study?",
    "hobbies": "What do you like to do in your free time?",
    "future_goal": "What is one goal you have for the future?",
}


def next_goal_question(state: dict[str, Any] | None) -> str:
    data = state or {}
    current = _trim(data.get("currentGoalId"))
    remaining = [_trim(item) for item in (data.get("goalsRemaining") or []) if _trim(item)]
    key = current if (current and current in remaining) or not remaining else remaining[0]
    if not key:
        key = current
    if key in _GOAL_QUESTIONS:
        return _GOAL_QUESTIONS[key]
    if key:
        return f"Can you tell me about {key.replace('_', ' ')}?"
    return "What else would you like to talk about?"
