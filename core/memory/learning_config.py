"""Configurable thresholds for learning-memory aggregation."""

from __future__ import annotations

from dataclasses import dataclass


LEARNING_MEMORY_CONFIG = {
    "minRecurringOccurrences": 2,
    "improvementEvidenceCount": 2,
    "resolutionEvidenceCount": 4,
    "maxRecurringMistakes": 50,
    "maxStrengths": 30,
    "maxPatterns": 30,
    "maxSourceConversationIdsPerSignal": 10,
    "maxImprovementAreas": 20,
    "maxSkillsPerCategory": 20,
    "minConfidence": 0.75,
    "assessmentOldWeight": 0.8,
    "assessmentNewWeight": 0.2,
}


@dataclass(frozen=True)
class LearningMemoryConfig:
    min_recurring_occurrences: int = 2
    improvement_evidence_count: int = 2
    resolution_evidence_count: int = 4
    max_recurring_mistakes: int = 50
    max_strengths: int = 30
    max_patterns: int = 30
    max_source_conversation_ids: int = 10
    max_improvement_areas: int = 20
    max_skills_per_category: int = 20
    min_confidence: float = 0.75
    assessment_old_weight: float = 0.8
    assessment_new_weight: float = 0.2

    @classmethod
    def from_mapping(cls, raw: dict | None = None) -> "LearningMemoryConfig":
        defaults = cls()
        if not raw:
            return defaults
        return cls(
            min_recurring_occurrences=int(
                raw.get("minRecurringOccurrences", defaults.min_recurring_occurrences)
            ),
            improvement_evidence_count=int(
                raw.get("improvementEvidenceCount", defaults.improvement_evidence_count)
            ),
            resolution_evidence_count=int(
                raw.get("resolutionEvidenceCount", defaults.resolution_evidence_count)
            ),
            max_recurring_mistakes=int(
                raw.get("maxRecurringMistakes", defaults.max_recurring_mistakes)
            ),
            max_strengths=int(raw.get("maxStrengths", defaults.max_strengths)),
            max_patterns=int(raw.get("maxPatterns", defaults.max_patterns)),
            max_source_conversation_ids=int(
                raw.get(
                    "maxSourceConversationIdsPerSignal",
                    defaults.max_source_conversation_ids,
                )
            ),
            max_improvement_areas=int(
                raw.get("maxImprovementAreas", defaults.max_improvement_areas)
            ),
            max_skills_per_category=int(
                raw.get("maxSkillsPerCategory", defaults.max_skills_per_category)
            ),
            min_confidence=float(raw.get("minConfidence", defaults.min_confidence)),
            assessment_old_weight=float(
                raw.get("assessmentOldWeight", defaults.assessment_old_weight)
            ),
            assessment_new_weight=float(
                raw.get("assessmentNewWeight", defaults.assessment_new_weight)
            ),
        )


DEFAULT_LEARNING_MEMORY_CONFIG = LearningMemoryConfig.from_mapping(
    LEARNING_MEMORY_CONFIG
)
