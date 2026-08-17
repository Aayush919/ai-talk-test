"""Live conversation + correction thresholds. Not a database."""

from __future__ import annotations

from dataclasses import dataclass


CONVERSATION_CONFIG = {
    "maxCorrectionsPerTurn": 1,
    "maxCorrectionsPerSession": 4,
    "minCorrectionConfidence": 0.85,
    "minSttConfidence": 0.70,
    "maxRecentQuestions": 48,
    "maxEntities": 16,
    "maxSpokenSentences": 3,
}


@dataclass(frozen=True)
class ConversationConfig:
    max_corrections_per_turn: int = 1
    max_corrections_per_session: int = 4
    min_correction_confidence: float = 0.85
    min_stt_confidence: float = 0.70
    max_recent_questions: int = 48
    max_entities: int = 16
    max_spoken_sentences: int = 3

    @classmethod
    def from_mapping(cls, raw: dict | None = None) -> "ConversationConfig":
        data = dict(CONVERSATION_CONFIG)
        if raw:
            data.update(raw)
        defaults = cls()
        return cls(
            max_corrections_per_turn=int(
                data.get("maxCorrectionsPerTurn", defaults.max_corrections_per_turn)
            ),
            max_corrections_per_session=int(
                data.get("maxCorrectionsPerSession", defaults.max_corrections_per_session)
            ),
            min_correction_confidence=float(
                data.get("minCorrectionConfidence", defaults.min_correction_confidence)
            ),
            min_stt_confidence=float(
                data.get("minSttConfidence", defaults.min_stt_confidence)
            ),
            max_recent_questions=int(
                data.get("maxRecentQuestions", defaults.max_recent_questions)
            ),
            max_entities=int(data.get("maxEntities", defaults.max_entities)),
            max_spoken_sentences=int(
                data.get("maxSpokenSentences", defaults.max_spoken_sentences)
            ),
        )


DEFAULT_CONVERSATION_CONFIG = ConversationConfig.from_mapping()
