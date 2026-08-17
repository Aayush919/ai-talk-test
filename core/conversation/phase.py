"""Conversation phase — guidance, not a forced script."""

from __future__ import annotations

from typing import Any

PHASES = (
    "START",
    "WARMUP",
    "EXPLORATION",
    "DEEPENING",
    "GOAL_COVERAGE",
    "TRANSITION",
    "CLOSING",
)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def next_phase(
    current: str | None,
    *,
    turn: int,
    engagement: str,
    user_intent: str,
    llm_phase: str | None,
    should_close: bool,
    should_transition: bool,
) -> str:
    if should_close or user_intent == "GOODBYE":
        return "CLOSING"
    suggested = _trim(llm_phase).upper()
    if suggested in PHASES and suggested != "START":
        if suggested == "CLOSING" and not should_close and user_intent != "GOODBYE":
            suggested = ""
        else:
            return suggested
    if turn <= 0:
        return "START"
    if turn == 1:
        return "WARMUP"
    if should_transition:
        return "TRANSITION"
    if engagement == "HIGH" and turn >= 3:
        return "DEEPENING"
    if turn >= 2:
        return "EXPLORATION"
    return _trim(current).upper() if _trim(current).upper() in PHASES else "WARMUP"
