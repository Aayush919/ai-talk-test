"""Call-only known facts. Regex only — no LLM, no Mongo, no Qdrant."""

from __future__ import annotations

import re
from typing import Any

_NAME = re.compile(
    r"\b(?:my name(?: is)?|this is|myself|i am|i'm)\s+"
    r"(?!living\b|from\b|doing\b|a\b|an\b|the\b|going\b|trying\b|stay(?:ing)?\b|"
    r"watch(?:ing)?\b|play(?:ing)?\b|read(?:ing)?\b)"
    r"([A-Za-z][A-Za-z']+(?:\s+[A-Za-z][A-Za-z']+)?)",
    re.I,
)
_LOCATION = re.compile(
    r"\b(?:(?:live|living|stay|staying) (?:in|at)|i'?m from|i am from|"
    r"belong(?:s)? to|grew up in)\s+([A-Za-z][A-Za-z]+)",
    re.I,
)
_WORK = re.compile(
    r"\b(?:i work as|i'm a|i am a|i am an|working profession)\s+([A-Za-z][A-Za-z\s]{2,30})?",
    re.I,
)
_NO_MUSIC = re.compile(
    r"\b(?:don'?t|do not|didn't|did not)\s+(?:listen(?:\s+to)?|like)\s+(?:any\s+)?(?:kind\s+of\s+)?music\b"
    r"|\bno music\b",
    re.I,
)
_SKIP_NAME = frozenset(
    {
        "living",
        "doing",
        "from",
        "okay",
        "yes",
        "yeah",
        "good",
        "fine",
        "here",
        "there",
        "student",
        "going",
        "trying",
        "using",
        "watching",
        "playing",
        "reading",
    }
)
_GOAL_FROM_FACT = {
    "name": ("name", "introduce_self"),
    "location": ("location",),
    "profession": ("education_or_work", "talk_about_work", "work", "role"),
    "education": ("education_or_work", "talk_about_work", "work_or_study_day"),
    "hobby": ("hobbies", "talk_about_hobbies"),
    "preference": ("hobbies", "talk_about_hobbies"),
    "background": ("talk_about_background",),
}
_MERGE_KEYS = frozenset({"hobby", "hobbies", "preference"})
_HOBBY_SKIP = re.compile(
    r"\bdo you remember\b|\bwhat kind of hobbies\b|\btell me (?:all )?my hobbies\b",
    re.I,
)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _title(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split() if part)


def _hobby_values(text: str) -> list[str]:
    lowered = text.lower()
    if _HOBBY_SKIP.search(lowered):
        return []
    found: list[str] = []

    def _add(value: str) -> None:
        item = " ".join(value.split())[:32].strip(" .,!")
        if not item or item in {"to", "a", "the", "my", "some"}:
            return
        if item.lower() not in {row.lower() for row in found}:
            found.append(item)

    if re.search(r"\bcricket\b", lowered):
        _add("cricket")
    if re.search(r"\bmovies?\b", lowered):
        _add("movies")
    if re.search(r"\b(?:read(?:ing)?(?:\s+books?)?|business books?)\b", lowered):
        _add("reading books")
    for match in re.finditer(
        r"\bi(?:'m| am)?\s+(?:like|love|enjoy|play(?:ing)?|watch(?:ing)?)\s+(?:to\s+)?(?:play\s+|watch\s+)?([a-z][a-z\s]{1,24})",
        lowered,
    ):
        _add(match.group(1).split(" with")[0].split(" and")[0].strip())
    return found


def extract_live_facts(text: str) -> list[dict[str, str]]:
    raw = _trim(text)
    if not raw:
        return []
    out: list[dict[str, str]] = []
    loc = _LOCATION.search(raw)
    if loc:
        city = loc.group(1)
        if city.lower() not in _SKIP_NAME:
            out.append({"key": "location", "value": _title(city)})
    name = _NAME.search(raw)
    if name:
        value = _title(name.group(1))
        first = value.split()[0].lower() if value else ""
        if first and first not in _SKIP_NAME:
            out.append({"key": "name", "value": value})
    work = _WORK.search(raw)
    if work and work.group(1):
        role = _trim(work.group(1)).split(".")[0].strip(" ,")
        if role and role.lower() not in _SKIP_NAME:
            out.append({"key": "profession", "value": role[:40]})
    lowered = raw.lower()
    if not any(row.get("key") == "profession" for row in out) and re.search(
        r"\bworking profession\b|\bi work\b", lowered
    ):
        out.append({"key": "profession", "value": "working professional"})
    if re.search(r"\bb\.?\s*tech\b|\bbtec\b|\bstudying\b|\bstudent\b|\bengineering\b", lowered):
        out.append({"key": "education", "value": "BTech" if "tech" in lowered or "btec" in lowered else "student"})
    hobbies = _hobby_values(raw)
    if hobbies:
        out.append({"key": "hobby", "value": ", ".join(hobbies)})
    if _NO_MUSIC.search(raw):
        out.append({"key": "preference", "value": "does not listen to music"})
    if re.search(r"\bgrew up\b|\bbackground\b|\bstudied\b", lowered):
        out.append({"key": "background", "value": "mentioned"})
    return out


def _merge_fact_value(key: str, existing: str, incoming: str) -> str:
    if key not in _MERGE_KEYS:
        return incoming or existing
    parts: list[str] = []
    for chunk in f"{existing}, {incoming}".split(","):
        item = chunk.strip()
        if not item or item.lower() == "mentioned":
            continue
        if item.lower() not in {row.lower() for row in parts}:
            parts.append(item)
    return ", ".join(parts[:6]) or incoming or existing


def merge_live_facts(existing: list[Any], incoming: list[dict[str, str]], *, limit: int = 12) -> list[dict[str, str]]:
    merged: dict[str, str] = {}
    for row in list(existing or []) + list(incoming or []):
        if not isinstance(row, dict):
            continue
        key = _trim(row.get("key"))
        value = _trim(row.get("value"))
        if key and value:
            merged[key] = _merge_fact_value(key, merged.get(key, ""), value)
    return [{"key": key, "value": value} for key, value in list(merged.items())[: max(1, limit)]]


def covered_goal_keys(
    facts: list[dict[str, str]],
    topic_goals: list[Any],
    already: list[Any] | None = None,
) -> list[str]:
    allowed: list[str] = []
    for goal in topic_goals or []:
        if isinstance(goal, dict) and goal.get("key"):
            allowed.append(str(goal["key"]))
        elif isinstance(goal, str) and goal:
            allowed.append(goal)
    covered = [str(item) for item in (already or []) if str(item) in set(allowed)]
    for row in facts or []:
        aliases = _GOAL_FROM_FACT.get(_trim(row.get("key")), ())
        for key in aliases:
            if key in allowed and key not in covered:
                covered.append(key)
    return covered


def advance_session_goals(
    *,
    topic_goals: list[Any],
    goals_completed: list[Any],
    goals_remaining: list[Any],
    facts: list[dict[str, str]],
) -> dict[str, Any]:
    keys: list[str] = []
    for goal in topic_goals or []:
        if isinstance(goal, dict) and goal.get("key"):
            keys.append(str(goal["key"]))
        elif isinstance(goal, str) and goal:
            keys.append(goal)
    completed = covered_goal_keys(facts, topic_goals, goals_completed)
    remaining = [
        key
        for key in (list(goals_remaining) if goals_remaining else keys)
        if key in set(keys) and key not in set(completed)
    ]
    if not remaining:
        remaining = [key for key in keys if key not in set(completed)]
    current = remaining[0] if remaining else (completed[-1] if completed else None)
    index = keys.index(current) if current in keys else None
    return {
        "goalsCompleted": completed,
        "goalsRemaining": remaining,
        "currentGoalId": current,
        "currentGoalIndex": index,
    }
