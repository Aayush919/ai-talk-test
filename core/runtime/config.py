"""Short-term conversation window — not a database."""

from __future__ import annotations

from dataclasses import dataclass


RUNTIME_CONFIG = {
    "maxRecentMessages": 12,
    "maxProfileFacts": 8,
    "maxLearningSignals": 5,
    "maxRelevantMemories": 5,
    "llmRetries": 0,
}


@dataclass(frozen=True)
class RuntimeConfig:
    max_recent_messages: int = 12
    max_profile_facts: int = 8
    max_learning_signals: int = 5
    max_relevant_memories: int = 5
    llm_retries: int = 0

    @classmethod
    def from_mapping(cls, raw: dict | None = None) -> "RuntimeConfig":
        defaults = cls()
        if not raw:
            return defaults
        return cls(
            max_recent_messages=int(
                raw.get("maxRecentMessages", defaults.max_recent_messages)
            ),
            max_profile_facts=int(
                raw.get("maxProfileFacts", defaults.max_profile_facts)
            ),
            max_learning_signals=int(
                raw.get("maxLearningSignals", defaults.max_learning_signals)
            ),
            max_relevant_memories=int(
                raw.get("maxRelevantMemories", defaults.max_relevant_memories)
            ),
            llm_retries=int(raw.get("llmRetries", defaults.llm_retries)),
        )


DEFAULT_RUNTIME_CONFIG = RuntimeConfig.from_mapping(RUNTIME_CONFIG)
