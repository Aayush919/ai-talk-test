"""Semantic memory thresholds — Qdrant is retrieval, not the source of truth."""

from __future__ import annotations

from dataclasses import dataclass


SEMANTIC_MEMORY_CONFIG = {
    "minConfidence": 0.70,
    "minImportance": 0.50,
    "similarityThreshold": 0.86,
    "tenantId": "talkengly",
    "maxCandidates": 8,
    "minContentLength": 24,
}

SEMANTIC_RETRIEVAL_CONFIG = {
    "topK": 5,
    "minSimilarityScore": 0.55,
    # Hashing cosine is much weaker than dense embeddings (live hits ~0.17–0.31).
    "hashingMinSimilarityScore": 0.12,
    "maxMemoriesForLLM": 5,
    "maxWeaknessesForLLM": 2,
    "retrieveOnTopicChange": True,
    "retrieveOnGoalChange": True,
    "retrieveEveryTurn": False,
    "mongoFallback": True,
    "improvingPenalty": 0.85,
    "resolvedPenalty": 0.25,
    "topicBoost": 0.08,
    "similarityWeight": 0.55,
    "importanceWeight": 0.20,
    "confidenceWeight": 0.15,
    "recencyWeight": 0.10,
}


@dataclass(frozen=True)
class SemanticMemoryConfig:
    min_confidence: float = 0.70
    min_importance: float = 0.50
    similarity_threshold: float = 0.86
    tenant_id: str = "talkengly"
    max_candidates: int = 8
    min_content_length: int = 24
    top_k: int = 5
    min_similarity_score: float = 0.55
    hashing_min_similarity_score: float = 0.12
    max_memories_for_llm: int = 5
    max_weaknesses_for_llm: int = 2
    retrieve_every_turn: bool = False
    retrieve_on_goal_change: bool = True
    retrieve_on_topic_change: bool = True
    mongo_fallback: bool = True
    improving_penalty: float = 0.85
    resolved_penalty: float = 0.25
    topic_boost: float = 0.08
    similarity_weight: float = 0.55
    importance_weight: float = 0.20
    confidence_weight: float = 0.15
    recency_weight: float = 0.10

    @classmethod
    def from_mapping(cls, raw: dict | None = None) -> "SemanticMemoryConfig":
        defaults = cls()
        data = dict(SEMANTIC_MEMORY_CONFIG)
        data.update(SEMANTIC_RETRIEVAL_CONFIG)
        if raw:
            data.update(raw)
        return cls(
            min_confidence=float(data.get("minConfidence", defaults.min_confidence)),
            min_importance=float(data.get("minImportance", defaults.min_importance)),
            similarity_threshold=float(
                data.get("similarityThreshold", defaults.similarity_threshold)
            ),
            tenant_id=str(data.get("tenantId", defaults.tenant_id) or "talkengly"),
            max_candidates=int(data.get("maxCandidates", defaults.max_candidates)),
            min_content_length=int(
                data.get("minContentLength", defaults.min_content_length)
            ),
            top_k=int(data.get("topK", defaults.top_k)),
            min_similarity_score=float(
                data.get("minSimilarityScore", defaults.min_similarity_score)
            ),
            hashing_min_similarity_score=float(
                data.get(
                    "hashingMinSimilarityScore",
                    defaults.hashing_min_similarity_score,
                )
            ),
            max_memories_for_llm=int(
                data.get("maxMemoriesForLLM", defaults.max_memories_for_llm)
            ),
            max_weaknesses_for_llm=int(
                data.get("maxWeaknessesForLLM", defaults.max_weaknesses_for_llm)
            ),
            retrieve_every_turn=bool(
                data.get("retrieveEveryTurn", defaults.retrieve_every_turn)
            ),
            retrieve_on_goal_change=bool(
                data.get("retrieveOnGoalChange", defaults.retrieve_on_goal_change)
            ),
            retrieve_on_topic_change=bool(
                data.get("retrieveOnTopicChange", defaults.retrieve_on_topic_change)
            ),
            mongo_fallback=bool(data.get("mongoFallback", defaults.mongo_fallback)),
            improving_penalty=float(
                data.get("improvingPenalty", defaults.improving_penalty)
            ),
            resolved_penalty=float(
                data.get("resolvedPenalty", defaults.resolved_penalty)
            ),
            topic_boost=float(data.get("topicBoost", defaults.topic_boost)),
            similarity_weight=float(
                data.get("similarityWeight", defaults.similarity_weight)
            ),
            importance_weight=float(
                data.get("importanceWeight", defaults.importance_weight)
            ),
            confidence_weight=float(
                data.get("confidenceWeight", defaults.confidence_weight)
            ),
            recency_weight=float(data.get("recencyWeight", defaults.recency_weight)),
        )

    def similarity_floor(self, embedding_model: str = "") -> float:
        """Dense embeddings keep 0.55. Local hashing needs a lower bar."""
        model = (embedding_model or "").strip().lower()
        if model.startswith("hash") or "hashing" in model:
            return float(self.hashing_min_similarity_score)
        return float(self.min_similarity_score)


DEFAULT_SEMANTIC_CONFIG = SemanticMemoryConfig.from_mapping()
