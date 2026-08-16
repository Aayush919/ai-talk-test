"""Pattern-based model sentences — no extra LLM on the voice path."""

from __future__ import annotations

import re

_I_USE_NAME = re.compile(r"\bi use\s+([a-z]{3,}(?:\s+[a-z]{2,})?)", re.I)
_GOING_NO_TO = re.compile(
    r"\b(?:i(?:'m| am)?\s+)?going\s+(?!to\b)(?:my |the )?(office|work|college|school|class|home)\b",
    re.I,
)
_AM_DOING_HABIT = re.compile(
    r"\bi(?:'m| am) doing\s+(yoga|coding|exercise|gym|cricket|football|homework)\b",
    re.I,
)
_AM_BASE = re.compile(
    r"\bi am\s+(go|eat|come|buy|see|get|take|play|work|live)\b",
    re.I,
)
_I_GOING = re.compile(r"\bi going\b", re.I)
_DOUBLE_I = re.compile(r"\bi i\s+(take|go|do|have|want|like)\b", re.I)
_GOING_MY = re.compile(r"\bgoing my (office|work|college|school)\b", re.I)


def model_for(user_text: str, *, name: str = "", lesson: str = "") -> str:
    """Return a short correct sentence the learner should say, or empty."""
    t = (user_text or "").strip()
    if not t:
        return ""

    m = _I_USE_NAME.search(t)
    if m:
        who = name or m.group(1).strip().title()
        return f"My name is {who}."

    m = _GOING_MY.search(t) or _GOING_NO_TO.search(t)
    if m:
        place = m.group(1).lower()
        if lesson == "daily_routine":
            return f"I go to the {place}."
        return f"I am going to the {place}."

    if lesson == "daily_routine":
        m = _AM_DOING_HABIT.search(t)
        if m and not re.search(r"\b(now|right now|currently)\b", t, re.I):
            return f"I do {m.group(1).lower()}."

    m = _AM_BASE.search(t)
    if m:
        return f"I {m.group(1).lower()}."

    if _I_GOING.search(t):
        rest = re.sub(r"^.*\bi going\s+", "", t, flags=re.I)
        rest = re.sub(r"[.?!].*$", "", rest).strip()
        if rest:
            if not rest.lower().startswith("to "):
                rest = f"to {rest}"
            return f"I am going {rest}.".replace("  ", " ")
        return "I am going."

    m = _DOUBLE_I.search(t)
    if m:
        verb = m.group(1).lower()
        rest = t[m.end() :].strip(" .,!?")
        rest = " ".join(rest.split()[:6])
        return f"I {verb} {rest}.".strip() if rest else f"I {verb}."

    return ""


def teach_line(wrong: str, model: str) -> str:
    """Spoken correction: you said X, we say Y, try it."""
    said = (wrong or "").strip().rstrip(".?!")
    if len(said) > 60:
        said = " ".join(said.split()[:10])
    return (
        f'You said "{said}". We don\'t say it like that. '
        f'Say it like this: "{model}" Try it.'
    )
