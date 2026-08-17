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


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm(value: Any) -> str:
    return " ".join(_trim(value).lower().split())


def empty_correction_state() -> dict[str, Any]:
    return {
        "correctionCandidates": [],
        "correctionsGivenThisTurn": 0,
        "correctionsGivenThisSession": 0,
        "recentCorrectionTypes": [],
        "recentOriginals": [],
    }


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
        current = dict(state.get("correctionState") or empty_correction_state())
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


def _only_informal_rewrite(original: str, corrected: str) -> bool:
    left = _norm(original)
    right = _norm(corrected)
    if left == right:
        return True
    expanded = left
    for informal, formal in _INFORMAL_PAIRS:
        expanded = re.sub(rf"\b{re.escape(informal)}\b", formal, expanded)
    return _norm(expanded) == right
