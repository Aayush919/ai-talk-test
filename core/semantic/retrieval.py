"""Memory retrieval pipeline — Qdrant search, Mongo validation, LangGraph injection.

Read-only on the live voice path. Mongo remains the source of truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core import call_log
from core.semantic.config import DEFAULT_SEMANTIC_CONFIG, SemanticMemoryConfig
from core.semantic.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from core.semantic.repository import InMemorySemanticRepository, SemanticMemoryRepository


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm(value: Any) -> str:
    return " ".join(_trim(value).lower().split())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compact_memory_context(memories: list[dict[str, Any]]) -> list[str]:
    """Short learner context for the LLM — content only, no Qdrant metadata."""
    lines: list[str] = []
    for row in memories:
        text = _trim(row.get("content"))
        if text:
            lines.append(text.rstrip("."))
    return lines


def build_retrieval_query(
    *,
    topic_title: str = "",
    current_goal: str = "",
    goal_description: str = "",
    learning_focus: str = "",
) -> str:
    topic = _trim(topic_title) or "English conversation"
    goal = _trim(goal_description) or _trim(current_goal).replace("_", " ")
    focus = _trim(learning_focus).replace("_", " ")
    parts = [
        "Relevant past experiences and English learning patterns",
        f"related to {topic}",
    ]
    if goal:
        parts.append(f"and practicing {goal}")
    if focus:
        parts.append(f"with a focus on {focus}")
    return " ".join(parts) + "."


def _active_learning_focus(learning: dict[str, Any] | None) -> str:
    for row in (learning or {}).get("recurringMistakes") or []:
        if not isinstance(row, dict):
            continue
        if _trim(row.get("status")).upper() == "ACTIVE":
            return _trim(row.get("skill"))
    return ""


def goal_description_from_topic(topic: dict[str, Any] | None, current_goal: str) -> str:
    wanted = _trim(current_goal)
    if not wanted:
        return ""
    for goal in (topic or {}).get("goals") or []:
        if isinstance(goal, dict) and _trim(goal.get("key")) == wanted:
            return _trim(goal.get("description"))
        if isinstance(goal, str) and goal == wanted:
            return wanted.replace("_", " ")
    return wanted.replace("_", " ")


@dataclass(frozen=True)
class RetrievalContext:
    tenant_id: str
    user_id: str
    query: str
    topic_id: str | None = None
    topic_title: str = ""
    topic_level: str | None = None
    current_goal: str = ""
    learning_focus: str = ""
    memory_types: tuple[str, ...] | None = None
    category: str | None = None
    skill: str | None = None
    status: str | None = None
    limit: int | None = None

    @property
    def fingerprint(self) -> str:
        return "|".join(
            [
                _trim(self.tenant_id),
                _trim(self.user_id),
                _trim(self.topic_id),
                _trim(self.current_goal),
                _trim(self.learning_focus),
            ]
        )


def build_retrieval_context(
    *,
    tenant_id: str,
    user_id: str,
    topic_id: str = "",
    topic_title: str = "",
    topic_level: str = "",
    current_goal: str = "",
    goal_description: str = "",
    learning: dict[str, Any] | None = None,
    learning_focus: str = "",
    memory_types: list[str] | None = None,
    limit: int | None = None,
) -> RetrievalContext:
    focus = _trim(learning_focus) or _active_learning_focus(learning)
    return RetrievalContext(
        tenant_id=_trim(tenant_id) or "talkengly",
        user_id=_trim(user_id),
        query=build_retrieval_query(
            topic_title=topic_title,
            current_goal=current_goal,
            goal_description=goal_description,
            learning_focus=focus,
        ),
        topic_id=_trim(topic_id) or None,
        topic_title=_trim(topic_title),
        topic_level=_trim(topic_level) or None,
        current_goal=_trim(current_goal),
        learning_focus=focus,
        memory_types=tuple(item for item in (memory_types or []) if item) or None,
        limit=limit,
    )


def _recency_score(value: Any, now: datetime) -> float:
    if value is None:
        return 0.4
    if isinstance(value, datetime):
        stamp = value
    else:
        try:
            stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return 0.4
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - stamp).total_seconds() / 86400.0)
    return 1.0 / (1.0 + days / 30.0)


def _slim(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "memoryId": row.get("memoryId"),
        "type": row.get("memoryType") or row.get("type"),
        "memoryType": row.get("memoryType") or row.get("type"),
        "content": row.get("content"),
        "confidence": row.get("confidence"),
        "status": row.get("status"),
        "scope": row.get("scope"),
        "skill": row.get("skill"),
        "score": row.get("score"),
        "relevanceScore": row.get("relevanceScore"),
    }


class MemoryRetrievalService:
    """Qdrant → Mongo validate → rank/filter → compact runtime memories."""

    def __init__(
        self,
        repo: Any,
        *,
        vectors: SemanticMemoryRepository | None = None,
        embeddings: EmbeddingProvider | None = None,
        config: SemanticMemoryConfig | dict | None = None,
        tenant_id: str = "talkengly",
        metrics: dict[str, float] | None = None,
    ) -> None:
        self.repo = repo
        self.vectors = vectors if vectors is not None else InMemorySemanticRepository()
        self.embeddings = embeddings or HashingEmbeddingProvider()
        if isinstance(config, SemanticMemoryConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = SemanticMemoryConfig.from_mapping(config)
        else:
            self.config = DEFAULT_SEMANTIC_CONFIG
        self.tenant_id = _trim(tenant_id) or self.config.tenant_id
        self.metrics = metrics if metrics is not None else {}

    def _similarity_floor(self) -> float:
        model = str(getattr(self.embeddings, "model", "") or "")
        return self.config.similarity_floor(model)

    def retrieve(self, context: RetrievalContext) -> list[dict[str, Any]]:
        tenant = _trim(context.tenant_id) or self.tenant_id
        uid = _trim(context.user_id)
        if not tenant or not uid:
            return []
        started = time.perf_counter()
        hits, source = self._search(context, tenant=tenant, user_id=uid)
        hits = self._merge_structured_hits(hits, context, tenant=tenant, user_id=uid)
        hits = self._enrich_from_mongo(hits, tenant=tenant, user_id=uid)
        out = self._rank_and_filter(hits, context, source=source)
        if not out and source == "qdrant":
            fallback = self._mongo_fallback(context, tenant=tenant, user_id=uid)
            fallback = self._enrich_from_mongo(fallback, tenant=tenant, user_id=uid)
            out = self._rank_and_filter(fallback, context, source="mongo")
            source = "mongo"
        self.metrics["retrieval_latency_ms"] = (time.perf_counter() - started) * 1000
        self.metrics["memory_retrieved"] = self.metrics.get("memory_retrieved", 0) + len(out)
        call_log.info(
            "MEMORY",
            "memory_retrieved",
            extra={"userId": uid, "count": len(out), "source": source},
        )
        return [_slim(row) for row in out]

    def retrieveForRuntime(
        self,
        *,
        tenant_id: str,
        user_id: str,
        topic_id: str = "",
        topic_title: str = "",
        topic_level: str = "",
        current_goal: str = "",
        goal_description: str = "",
        learning: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        context = build_retrieval_context(
            tenant_id=tenant_id or self.tenant_id,
            user_id=user_id,
            topic_id=topic_id,
            topic_title=topic_title,
            topic_level=topic_level,
            current_goal=current_goal,
            goal_description=goal_description,
            learning=learning,
            limit=limit,
        )
        return self.retrieve(context), context.fingerprint

    def _search(
        self, context: RetrievalContext, *, tenant: str, user_id: str
    ) -> tuple[list[dict[str, Any]], str]:
        available = getattr(self.vectors, "available", None)
        online = available() if callable(available) else True
        if not online:
            return self._mongo_fallback(context, tenant=tenant, user_id=user_id), "mongo"
        extra_filters: dict[str, Any] = {}
        if context.category:
            extra_filters["category"] = context.category
        if context.skill:
            extra_filters["skill"] = context.skill
        if context.status:
            extra_filters["status"] = context.status
        top_k = int(context.limit or self.config.top_k)
        try:
            embed_started = time.perf_counter()
            vector = self.embeddings.embed(context.query)
            self.metrics["embedding_latency_ms"] = (time.perf_counter() - embed_started) * 1000
            search_started = time.perf_counter()
            points = self.vectors.searchMemories(
                vector=vector,
                tenant_id=tenant,
                user_id=user_id,
                limit=max(top_k * 3, top_k),
                filters=extra_filters or None,
            )
            self.metrics["qdrant_search_latency_ms"] = (
                time.perf_counter() - search_started
            ) * 1000
        except Exception as exc:  # noqa: BLE001
            self.metrics["qdrant_failures"] = self.metrics.get("qdrant_failures", 0) + 1
            call_log.error(
                "MEMORY",
                "QDRANT_RETRIEVAL_FAILED",
                extra={"detail": str(exc)[:240]},
            )
            call_log.warn("MEMORY", "memory_retrieval_failed")
            return self._mongo_fallback(context, tenant=tenant, user_id=user_id), "mongo"
        hits: list[dict[str, Any]] = []
        for point in points:
            payload = dict(point.get("payload") or {})
            if str(payload.get("tenantId") or "") != tenant:
                continue
            if str(payload.get("userId") or "") != user_id:
                continue
            hits.append(
                {
                    "id": point.get("id"),
                    "similarity": float(point.get("score") or 0),
                    "payload": payload,
                    "source": "qdrant",
                }
            )
        return hits, "qdrant"

    def _merge_structured_hits(
        self,
        hits: list[dict[str, Any]],
        context: RetrievalContext,
        *,
        tenant: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        focus = _norm(context.learning_focus)
        topic = _trim(context.topic_id)
        if not focus and not topic:
            return hits
        lister = getattr(self.repo, "list_memory_metadata", None)
        if not callable(lister):
            return hits
        seen = {
            _trim((hit.get("payload") or {}).get("memoryId") or hit.get("id"))
            for hit in hits
        }
        floor = self._similarity_floor()
        for row in lister(tenant_id=tenant, user_id=user_id) or []:
            if not isinstance(row, dict):
                continue
            memory_id = _trim(row.get("memoryId"))
            if not memory_id or memory_id in seen:
                continue
            skill_match = focus and _norm(row.get("skill")) == focus
            topic_match = topic and str(row.get("topicId") or "") == topic
            if not skill_match and not topic_match:
                continue
            hits.append(
                {
                    "id": row.get("qdrantPointId") or memory_id,
                    "similarity": floor,
                    "payload": dict(row),
                    "source": "mongo_structured",
                }
            )
            seen.add(memory_id)
        return hits

    def _mongo_fallback(
        self, context: RetrievalContext, *, tenant: str, user_id: str
    ) -> list[dict[str, Any]]:
        if not self.config.mongo_fallback:
            return []
        lister = getattr(self.repo, "list_memory_metadata", None)
        if not callable(lister):
            return []
        rows = lister(tenant_id=tenant, user_id=user_id) or []
        hits: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("tenantId") or "") != tenant or str(row.get("userId") or "") != user_id:
                continue
            hits.append(
                {
                    "id": row.get("qdrantPointId") or row.get("memoryId"),
                    "similarity": 0.0,
                    "payload": dict(row),
                    "source": "mongo",
                }
            )
        return hits

    def _enrich_from_mongo(
        self, hits: list[dict[str, Any]], *, tenant: str, user_id: str
    ) -> list[dict[str, Any]]:
        finder = getattr(self.repo, "find_memory_metadata", None)
        if not callable(finder):
            return hits
        out: list[dict[str, Any]] = []
        for hit in hits:
            payload = dict(hit.get("payload") or {})
            memory_id = _trim(payload.get("memoryId") or hit.get("id"))
            doc = finder(memory_id) if memory_id else None
            if not isinstance(doc, dict):
                if hit.get("source") == "mongo":
                    out.append(hit)
                continue
            if str(doc.get("tenantId") or "") != tenant or str(doc.get("userId") or "") != user_id:
                continue
            if _trim(doc.get("status")).upper() == "ARCHIVED":
                continue
            merged = dict(payload)
            for key in (
                "memoryId",
                "memoryType",
                "category",
                "skill",
                "content",
                "scope",
                "topicId",
                "topicLevel",
                "importance",
                "confidence",
                "status",
                "frequency",
                "lastSeenAt",
                "updatedAt",
            ):
                if doc.get(key) is not None:
                    merged[key] = doc.get(key)
            hit = dict(hit)
            hit["payload"] = merged
            out.append(hit)
        return out

    def _rank_and_filter(
        self,
        hits: list[dict[str, Any]],
        context: RetrievalContext,
        *,
        source: str,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        cap = min(int(context.limit or cfg.top_k), cfg.max_memories_for_llm)
        allowed_types = {item.upper() for item in (context.memory_types or ()) if item}
        now = _utc_now()
        ranked: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.get("payload") or {}
            memory_type = _trim(payload.get("memoryType")).upper()
            if allowed_types and memory_type not in allowed_types:
                continue
            similarity = float(hit.get("similarity") or 0)
            hit_source = str(hit.get("source") or source)
            if hit_source == "qdrant" and similarity < self._similarity_floor():
                continue
            status = _trim(payload.get("status")).upper() or "ACTIVE"
            if status == "ARCHIVED":
                continue
            importance = float(payload.get("importance") or 0)
            confidence = float(payload.get("confidence") or 0)
            recency = _recency_score(
                payload.get("lastSeenAt") or payload.get("updatedAt"), now
            )
            score = (
                cfg.similarity_weight * (similarity if source != "mongo" else 0.0)
                + cfg.importance_weight * importance
                + cfg.confidence_weight * confidence
                + cfg.recency_weight * recency
            )
            if source == "mongo":
                score = (
                    cfg.importance_weight * importance
                    + cfg.confidence_weight * confidence
                    + cfg.recency_weight * recency
                )
            if status == "RESOLVED":
                score *= cfg.resolved_penalty
            elif status == "IMPROVING":
                score *= cfg.improving_penalty
            if context.topic_id and str(payload.get("topicId") or "") == str(context.topic_id):
                score += cfg.topic_boost
            ranked.append(
                {
                    "memoryId": payload.get("memoryId") or hit.get("id"),
                    "memoryType": memory_type or payload.get("memoryType"),
                    "type": memory_type or payload.get("memoryType"),
                    "content": payload.get("content"),
                    "confidence": confidence,
                    "importance": importance,
                    "status": status,
                    "scope": payload.get("scope") or "GLOBAL",
                    "skill": payload.get("skill"),
                    "score": round(score, 4),
                    "relevanceScore": round(similarity, 4),
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        topic_hits = [row for row in ranked if row.get("scope") == "TOPIC"]
        global_hits = [row for row in ranked if row.get("scope") != "TOPIC"]
        merged = topic_hits + global_hits
        seen: set[str] = set()
        weaknesses = 0
        out: list[dict[str, Any]] = []
        for row in merged:
            key = _norm(row.get("content"))
            if not key or key in seen:
                continue
            memory_type = _trim(row.get("memoryType")).upper()
            if memory_type == "LEARNING_WEAKNESS":
                if weaknesses >= cfg.max_weaknesses_for_llm:
                    continue
                weaknesses += 1
            seen.add(key)
            out.append(row)
            if len(out) >= cap:
                break
        return out
