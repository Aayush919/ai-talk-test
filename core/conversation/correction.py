"""Selective live correction — backend gated. Never writes learning_memory."""

from __future__ import annotations

import re
from typing import Any

from core.conversation.config import ConversationConfig, DEFAULT_CONVERSATION_CONFIG

_INFORMAL_PAIRS = (
    ("wanna", "want to"),
    ("gonna", "going to"),
    ("gotta", "got to"),
    ("yeah", "yes"),
    ("ok", "okay"),
)
CORRECTION_TYPES = frozenset(
    {
        "GRAMMAR",
        "WORD_CHOICE",
        "SENTENCE_STRUCTURE",
        "VOCABULARY",
        "PRONUNCIATION",
        "FLUENCY",
        "NATURALNESS",
    }
)
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirty": "30",
}
_STOP = frozenset(
    "i a an the to at and then please usually around about am is are was were of in on".split()
)
_REFUSE_REPEAT = re.compile(
    r"^\s*(no|nope|nah)\s*[.!]?\s*$|"
    r"\b(i don't want to|don't want to repeat|not now|skip that|later)\b",
    re.I,
)
_ASK_REPEAT = re.compile(
    r"\b(?:please )?repeat(?: it)?(?: once)?\b|\btry saying\b|\bsay it (?:once|again)\b",
    re.I,
)
_QUOTED = re.compile(r"['\"]([^'\"]{6,80})['\"]")
_SAY_LINE = re.compile(
    r"(?:you can say|try this|try saying|a more natural way is|"
    r"let's say it this way|say it like this)[,:]?\s*['\"]?([^'\"\n]+)",
    re.I,
)
_COOLDOWN_TURNS = 2


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm(value: Any) -> str:
    return " ".join(_trim(value).lower().split())


def empty_correction_state() -> dict[str, Any]:
    return {
        "status": "idle",
        "originalText": None,
        "correctedText": None,
        "errorType": None,
        "confidence": 0.0,
        "attempts": 0,
        "lastCorrectionAt": None,
        "clarifyGiven": False,
        "lastOutcome": None,
        "correctionCandidates": [],
        "correctionsGivenThisTurn": 0,
        "correctionsGivenThisSession": 0,
        "recentCorrectionTypes": [],
        "recentOriginals": [],
    }


def _fold_speech(value: Any) -> str:
    text = _norm(value).replace(":", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    for word, digit in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    return " ".join(text.split())


def _content_tokens(text: str) -> list[str]:
    return [part for part in _fold_speech(text).split() if part not in _STOP]


def repeat_accepted(user_text: str, target: str) -> bool:
    """Meaning + target structure, not exact string match."""
    if not _trim(user_text) or not _trim(target):
        return False
    wanted = _content_tokens(target)
    got = _content_tokens(user_text)
    if not wanted or not got:
        return _fold_speech(user_text) == _fold_speech(target)
    stems_got = {token.rstrip("ing").rstrip("ed") for token in got}
    missing = [
        token
        for token in wanted
        if token not in got and token.rstrip("ing").rstrip("ed") not in stems_got
    ]
    return len(missing) <= max(0, len(wanted) // 4) and len(got) >= 2


def extract_presented_correction(reply: str) -> str | None:
    raw = _trim(reply)
    if not raw or not _ASK_REPEAT.search(raw):
        return None
    quoted = _QUOTED.findall(raw)
    if quoted:
        return _trim(quoted[0]).strip(" .,!")
    match = _SAY_LINE.search(raw)
    if match:
        return _trim(match.group(1)).split("Please")[0].strip(" .,!")
    return None


def correction_prompt_lines(
    state: dict[str, Any] | None,
    *,
    turn: int = 0,
) -> list[str]:
    data = state or empty_correction_state()
    status = _trim(data.get("status")) or "idle"
    outcome = _trim(data.get("lastOutcome"))
    target = _trim(data.get("correctedText"))
    lines: list[str] = []
    if outcome == "accepted":
        lines.append(
            "They repeated the corrected sentence (or close enough). "
            "Praise briefly and continue the topic. Do not recast again."
        )
        return lines
    if outcome == "dismissed":
        lines.append(
            "They did not repeat. Do not ask them to repeat. Continue the conversation."
        )
        return lines
    if outcome == "clarify" or (status == "awaiting_repeat" and data.get("clarifyGiven")):
        lines.append(
            "They did not understand the correction. "
            "Say once: try saying the sentence once. Then wait. Do not lecture."
        )
        if target:
            lines.append(f"Target sentence: {target}")
        return lines
    if status == "awaiting_repeat" and target:
        lines.append(f"Correction in progress. Target: '{target}'")
        lines.append(
            "If they said it (even a bit differently), praise and continue. "
            "If they refused or talked about something else, do not ask to repeat."
        )
        return lines
    session = int(data.get("correctionsGivenThisSession") or 0)
    last_at = data.get("lastCorrectionAt")
    if session >= 4:
        lines.append("Correction cooldown. Do not correct this turn. Just talk.")
        return lines
    if last_at is not None and int(turn or 0) - int(last_at) <= _COOLDOWN_TURNS:
        lines.append("Correction cooldown. Do not correct this turn. Just talk.")
        return lines
    lines.append(
        "If there is one clear useful mistake, recast it in this same reply: "
        "Nice! You can say, 'corrected sentence.' Please repeat it once. "
        "Otherwise just talk. Never correct wanna/gonna/yeah or likely STT junk."
    )
    return lines


def resolve_awaiting_repeat(
    state: dict[str, Any] | None,
    *,
    user_text: str,
    user_intent: str,
) -> dict[str, Any]:
    current = {**empty_correction_state(), **(state or {})}
    if _trim(current.get("status")) != "awaiting_repeat":
        current["lastOutcome"] = None
        return current
    target = _trim(current.get("correctedText"))
    intent = _trim(user_intent)
    if intent == "REFUSAL" or _REFUSE_REPEAT.search(_trim(user_text)):
        current["status"] = "dismissed"
        current["lastOutcome"] = "dismissed"
        return current
    if intent == "CONFUSION":
        if current.get("clarifyGiven"):
            current["status"] = "dismissed"
            current["lastOutcome"] = "dismissed"
        else:
            current["clarifyGiven"] = True
            current["lastOutcome"] = "clarify"
            current["status"] = "awaiting_repeat"
        return current
    if repeat_accepted(user_text, target):
        current["status"] = "completed"
        current["lastOutcome"] = "accepted"
        return current
    current["status"] = "dismissed"
    current["lastOutcome"] = "dismissed"
    return current


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_correction(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    if raw.get("shouldCorrect") is False:
        return None
    original = _trim(raw.get("original") or raw.get("originalText"))
    corrected = _trim(raw.get("corrected") or raw.get("correctedText"))
    if not original or not corrected:
        return None
    if _norm(original) == _norm(corrected):
        return None
    kind = _trim(raw.get("type") or raw.get("correctionType")).upper() or "GRAMMAR"
    if kind == "NATURALNESS":
        kind = "WORD_CHOICE"
    if kind not in CORRECTION_TYPES:
        kind = "GRAMMAR"
    confidence = _as_float(raw.get("confidence"))
    if confidence is None:
        confidence = 0.9 if raw.get("shouldCorrect") else 0.0
    severity = _trim(raw.get("severity")).upper() or "MEDIUM"
    return {
        "type": kind,
        "original": original,
        "corrected": corrected,
        "explanation": _trim(raw.get("explanation") or raw.get("concept")),
        "severity": severity,
        "confidence": max(0.0, min(1.0, confidence)),
        "concept": _trim(raw.get("concept")),
    }


class CorrectionService:
    def __init__(self, config: ConversationConfig | None = None) -> None:
        self.config = config or DEFAULT_CONVERSATION_CONFIG

    def filter_live_correction(
        self,
        raw: Any,
        *,
        correction_state: dict[str, Any] | None,
        user_intent: str = "ANSWER",
        stt_confidence: float | None = None,
        pronunciation_evidence: bool = False,
    ) -> dict[str, Any] | None:
        candidate = normalize_correction(raw)
        if candidate is None:
            return None
        state = correction_state or empty_correction_state()
        if _trim(state.get("status")) == "awaiting_repeat" and user_intent != "CORRECTION_REQUEST":
            return None
        requested = user_intent == "CORRECTION_REQUEST"
        if candidate["type"] == "PRONUNCIATION" and not pronunciation_evidence:
            return None
        if stt_confidence is not None and stt_confidence < self.config.min_stt_confidence:
            return None
        if candidate["confidence"] < self.config.min_correction_confidence and not requested:
            return None
        if candidate["severity"] == "LOW" and not requested:
            return None
        if _only_informal_rewrite(candidate["original"], candidate["corrected"]):
            return None
        given_turn = int(state.get("correctionsGivenThisTurn") or 0)
        given_session = int(state.get("correctionsGivenThisSession") or 0)
        if given_turn >= self.config.max_corrections_per_turn:
            return None
        if (
            given_session >= self.config.max_corrections_per_session
            and not requested
        ):
            return None
        originals = [_norm(item) for item in (state.get("recentOriginals") or [])]
        if _norm(candidate["original"]) in originals and not requested:
            candidate["repeat"] = True
        return candidate

    def apply_to_state(
        self,
        state: dict[str, Any],
        candidate: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current = {**empty_correction_state(), **(state.get("correctionState") or {})}
        current["correctionsGivenThisTurn"] = 0
        if candidate is None:
            current["correctionCandidates"] = list(current.get("correctionCandidates") or [])
            return current
        current["correctionsGivenThisTurn"] = 1
        current["correctionsGivenThisSession"] = int(
            current.get("correctionsGivenThisSession") or 0
        ) + 1
        types = list(current.get("recentCorrectionTypes") or [])
        types.append(candidate["type"])
        current["recentCorrectionTypes"] = types[-8:]
        originals = list(current.get("recentOriginals") or [])
        originals.append(candidate["original"])
        current["recentOriginals"] = originals[-8:]
        if candidate.get("repeat"):
            current["correctionCandidates"] = list(current.get("correctionCandidates") or [])
        else:
            pending = list(current.get("correctionCandidates") or [])
            pending.append(
                {
                    "type": candidate["type"],
                    "original": candidate["original"],
                    "corrected": candidate["corrected"],
                    "concept": candidate.get("concept") or candidate.get("explanation"),
                    "severity": candidate["severity"],
                }
            )
            current["correctionCandidates"] = pending[-6:]
        return current

    def after_spoken_reply(
        self,
        correction_state: dict[str, Any] | None,
        *,
        reply: str,
        user_text: str,
        user_intent: str,
        turn: int,
    ) -> dict[str, Any]:
        current = {**empty_correction_state(), **(correction_state or {})}
        outcome = _trim(current.get("lastOutcome"))
        if outcome in {"accepted", "dismissed"}:
            current["status"] = "idle"
            current["originalText"] = None
            current["correctedText"] = None
            current["clarifyGiven"] = False
            current["attempts"] = 0
            current["lastOutcome"] = None
            return current
        if _trim(current.get("status")) == "awaiting_repeat":
            return current
        presented = extract_presented_correction(reply)
        if not presented:
            return current
        if user_intent in {"MEMORY_PROBE", "REPEAT_COMPLAINT", "CONFUSION", "GOODBYE"}:
            return current
        session = int(current.get("correctionsGivenThisSession") or 0)
        if (
            session >= self.config.max_corrections_per_session
            and user_intent != "CORRECTION_REQUEST"
        ):
            return current
        last_at = current.get("lastCorrectionAt")
        if last_at is not None and int(turn or 0) - int(last_at) <= _COOLDOWN_TURNS:
            return current
        originals = [_norm(item) for item in (current.get("recentOriginals") or [])]
        if _norm(user_text) in originals and user_intent != "CORRECTION_REQUEST":
            return current
        current["status"] = "awaiting_repeat"
        current["originalText"] = _trim(user_text) or None
        current["correctedText"] = presented
        current["attempts"] = 1
        current["clarifyGiven"] = False
        current["lastCorrectionAt"] = turn
        current["lastOutcome"] = None
        if int(current.get("correctionsGivenThisTurn") or 0) == 0:
            current["correctionsGivenThisTurn"] = 1
            current["correctionsGivenThisSession"] = session + 1
        originals.append(_trim(user_text))
        current["recentOriginals"] = [item for item in originals if item][-8:]
        return current


def _only_informal_rewrite(original: str, corrected: str) -> bool:
    left = _norm(original)
    right = _norm(corrected)
    if left == right:
        return True
    expanded = left
    for informal, formal in _INFORMAL_PAIRS:
        expanded = re.sub(rf"\b{re.escape(informal)}\b", formal, expanded)
    return _norm(expanded) == right
