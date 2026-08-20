"""Parse and validate a single live AI turn. TTS gets text only."""

from __future__ import annotations

import re
from typing import Any

from core.conversations.summary_service import parse_json_object

FORBIDDEN_DECISION_KEYS = frozenset(
    {
        "databaseOperation",
        "topicProgress",
        "userId",
        "conversationId",
        "goalsCompleted",
        "goalsRemaining",
    }
)

SPEAK_INTENTS = frozenset(
    {
        "ANSWER",
        "FOLLOW_UP",
        "CORRECTION",
        "CLARIFICATION",
        "OFF_TOPIC",
        "CLOSING",
        "OPEN_TOPIC",
        "DEEPEN",
        "TRANSITION",
        "CLARIFY",
        "ENCOURAGE",
        "CLOSE",
    }
)
_INTENT_MAP = {
    "OPEN_TOPIC": "FOLLOW_UP",
    "DEEPEN": "FOLLOW_UP",
    "ENCOURAGE": "FOLLOW_UP",
    "CLARIFY": "CLARIFICATION",
    "CLOSE": "CLOSING",
    "TRANSITION": "FOLLOW_UP",
}
_QUESTION = re.compile(r"[^.!?]*\?")


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_question(text: str) -> str | None:
    matches = _QUESTION.findall(text or "")
    if not matches:
        return None
    question = _trim(matches[-1])
    return question or None


def question_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _trim(text).lower()).strip()


_QUESTION_STOP = frozenset(
    """
    do you what when where why how the a an to in on of your usually can
    tell me please about a bit little more also and or so okay
    """.split()
)


def is_similar_question(left: str, right: str) -> bool:
    a = {part for part in question_key(left).split() if part not in _QUESTION_STOP}
    b = {part for part in question_key(right).split() if part not in _QUESTION_STOP}
    if not a or not b:
        return bool(question_key(left)) and question_key(left) == question_key(right)
    overlap = len(a & b)
    if overlap >= 2:
        return True
    return overlap / len(a | b) >= 0.5


def rewrite_repeated_question(reply: str, old_question: str, new_question: str) -> str:
    spoken = _trim(reply)
    previous = _trim(old_question)
    nxt = _trim(new_question)
    if not nxt:
        return spoken
    if previous and previous in spoken:
        return spoken.replace(previous, nxt, 1)
    stripped = re.sub(r"[^.!?]*\?\s*$", "", spoken).strip()
    if stripped:
        return f"{stripped} {nxt}"
    return nxt


def generateNextQuestion(context: dict[str, Any], decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Use the already-generated turn. Never starts a second LLM call."""
    decision = decision or {}
    question = _trim(decision.get("question")) or extract_question(
        _trim(decision.get("text") or decision.get("response"))
    )
    return {
        "question": question,
        "purpose": _trim(decision.get("purpose") or context.get("conversationPhase") or "follow_up"),
        "expectedArea": _trim(decision.get("expectedArea") or context.get("currentGoalId")),
    }


def parse_ai_response(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        try:
            raw = parse_json_object(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(raw, dict):
        return None
    if any(key in raw for key in FORBIDDEN_DECISION_KEYS):
        return None
    text = _trim(raw.get("text") or raw.get("response"))
    if not text:
        return None
    raw_intent = _trim(raw.get("intent")).upper() or "FOLLOW_UP"
    if raw_intent not in SPEAK_INTENTS:
        raw_intent = "FOLLOW_UP"
    public_intent = _INTENT_MAP.get(raw_intent, raw_intent)
    question = _trim(raw.get("question")) or extract_question(text)
    evidence = raw.get("goalEvidence") if isinstance(raw.get("goalEvidence"), dict) else None
    if evidence is not None:
        evidence = {
            "goalId": _trim(evidence.get("goalId") or evidence.get("goal_id")),
            "coveredAreas": [
                str(item)
                for item in (evidence.get("coveredAreas") or evidence.get("areasCovered") or [])
                if str(item).strip()
            ],
            "remainingAreas": [
                str(item)
                for item in (evidence.get("remainingAreas") or evidence.get("areasRemaining") or [])
                if str(item).strip()
            ],
            "completionConfidence": _clamp(evidence.get("completionConfidence") or evidence.get("confidence")),
        }
    return {
        "text": text,
        "response": text,
        "intent": public_intent,
        "rawIntent": raw_intent,
        "question": question,
        "purpose": _trim(raw.get("purpose")),
        "expectedArea": _trim(raw.get("expectedArea")),
        "correction": raw.get("correction") if isinstance(raw.get("correction"), dict) else None,
        "goalEvidence": evidence,
        "conversationPhase": _trim(raw.get("conversationPhase")).upper() or None,
        "shouldContinue": _as_bool(raw.get("shouldContinue"), default=True),
        "shouldTransition": _as_bool(raw.get("shouldTransition"), default=False),
        "targetGoalId": _trim(raw.get("targetGoalId") or raw.get("targetGoal")) or None,
        "followUpNeeded": _as_bool(raw.get("followUpNeeded"), default=bool(question)),
        "questionType": _trim(raw.get("questionType")).upper() or None,
    }


def spoken_text_only(decision: dict[str, Any] | None) -> str:
    return _trim((decision or {}).get("text") or (decision or {}).get("response"))


def coerce_spoken_text(raw: Any) -> str:
    """Accept plain speech, or recover `text` if the model still returned JSON."""
    if isinstance(raw, dict):
        return _trim(raw.get("text") or raw.get("response"))
    text = _trim(raw)
    if not text:
        return ""
    if text.startswith("{") or text.startswith("```"):
        try:
            data = parse_json_object(text)
        except (ValueError, TypeError):
            return text
        if isinstance(data, dict):
            return _trim(data.get("text") or data.get("response")) or text
    return text


def decision_from_spoken(text: str, *, user_intent: str = "ANSWER") -> dict[str, Any]:
    spoken = _trim(text)
    question = extract_question(spoken)
    intent = "FOLLOW_UP"
    mapped = _trim(user_intent).upper()
    if mapped == "GOODBYE":
        intent = "CLOSING"
    elif mapped == "CORRECTION_REQUEST":
        intent = "CORRECTION"
    elif mapped == "QUESTION":
        intent = "ANSWER"
    elif mapped == "OFF_TOPIC":
        intent = "OFF_TOPIC"
    return {
        "text": spoken,
        "response": spoken,
        "intent": intent,
        "question": question,
        "purpose": "",
        "expectedArea": "",
        "correction": None,
        "goalEvidence": None,
        "conversationPhase": None,
        "shouldContinue": mapped != "GOODBYE",
        "shouldTransition": False,
        "targetGoalId": None,
        "followUpNeeded": bool(question),
        "questionType": None,
    }


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
