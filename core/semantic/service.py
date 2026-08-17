"""Semantic memories — Mongo first, Qdrant index, tenant-safe retrieval."""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from core import call_log
from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationNotCompleted,
    ConversationNotFound,
    SummaryNotFound,
)
from core.conversations.summary_service import parse_json_object
from core.db.schema import QDRANT_MEMORY_TYPES
from core.semantic.config import DEFAULT_SEMANTIC_CONFIG, SemanticMemoryConfig
from core.semantic.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from core.semantic.prompt import (
    SEMANTIC_EXTRACTION_SYSTEM_PROMPT,
    build_semantic_extraction_prompt,
)
from core.semantic.repository import InMemorySemanticRepository, SemanticMemoryRepository
from core.semantic.retrieval import (
    MemoryRetrievalService,
    RetrievalContext,
    build_retrieval_context,
    build_retrieval_query,
    compact_memory_context,
)

COMPLETED = "COMPLETED"
MEMORY_TYPES = frozenset(QDRANT_MEMORY_TYPES)
SOURCE_TYPES = frozenset({"conversation_summary", "learning_memory", "profile_memory"})
SCOPES = frozenset({"GLOBAL", "TOPIC"})
STATUSES = frozenset({"ACTIVE", "IMPROVING", "RESOLVED", "ARCHIVED"})
SKIP_PROFILE_KEYS = frozenset({"name"})
_POLLUTION = re.compile(
    r"\b(hello|hi|yes|yeah|ok|okay|paused?|said hello|goals? completed)\b"
    r"|\d+\s*/\s*\d+",
    re.I,
)
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
CATEGORY_FROM_TYPE = {
    "PROFILE_FACT": "profile",
    "LEARNING_PATTERN": "learning",
    "LEARNING_WEAKNESS": "learning",
    "LEARNING_STRENGTH": "learning",
    "EXPERIENCE": "experience",
    "PREFERENCE": "preference",
    "CONVERSATION_MEMORY": "conversation",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm(value: Any) -> str:
    return " ".join(_trim(value).lower().split())


def identity_key(memory_type: str, skill: str, content: str) -> str:
    skill_slug = re.sub(r"[^a-z0-9]+", "_", _norm(skill)).strip("_")
    if skill_slug:
        return f"{memory_type}|{skill_slug}"
    tokens = _norm(content).split()[:8]
    return f"{memory_type}|{'_'.join(tokens)}"


def memory_ids(*, tenant_id: str, user_id: str, identity: str) -> tuple[str, str]:
    seed = f"{tenant_id}|{user_id}|{identity}"
    point_id = str(uuid.uuid5(_NAMESPACE, seed))
    return point_id, point_id


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class SemanticAnalyzer(Protocol):
    def analyze_json(self, *, system: str, user: str) -> dict[str, Any]: ...


class SemanticMemoryService:
    def __init__(
        self,
        repo: Any,
        *,
        vectors: SemanticMemoryRepository | None = None,
        embeddings: EmbeddingProvider | None = None,
        analyzer: SemanticAnalyzer | None = None,
        config: SemanticMemoryConfig | dict | None = None,
        tenant_id: str = "talkengly",
        embedding_model: str = "hashing-v1",
        embedding_version: str = "v1",
    ) -> None:
        self.repo = repo
        self.vectors = vectors if vectors is not None else InMemorySemanticRepository()
        self.embeddings = embeddings or HashingEmbeddingProvider()
        self.analyzer = analyzer
        if isinstance(config, SemanticMemoryConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = SemanticMemoryConfig.from_mapping(config)
        else:
            self.config = DEFAULT_SEMANTIC_CONFIG
        self.tenant_id = _trim(tenant_id) or self.config.tenant_id
        self.embedding_model = embedding_model or self.embeddings.model
        self.embedding_version = embedding_version or getattr(self.embeddings, "version", "v1")
        self.metrics: dict[str, float] = {
            "memory_created": 0,
            "memory_updated": 0,
            "memory_indexed": 0,
            "memory_retrieved": 0,
            "memory_deduplicated": 0,
            "qdrant_failures": 0,
            "embedding_failures": 0,
            "retrieval_latency_ms": 0,
            "qdrant_search_latency_ms": 0,
            "embedding_latency_ms": 0,
        }
        self.retrieval = MemoryRetrievalService(
            self.repo,
            vectors=self.vectors,
            embeddings=self.embeddings,
            config=self.config,
            tenant_id=self.tenant_id,
            metrics=self.metrics,
        )

    def qdrantHealthCheck(self) -> str:
        check = getattr(self.vectors, "healthCheck", None)
        if callable(check):
            return check()
        return "QDRANT_UNAVAILABLE"

    def extractSemanticMemories(
        self,
        summary: dict[str, Any],
        *,
        learning: dict[str, Any] | None = None,
        existing: list[dict[str, Any]] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self._from_learning(learning, source_id=_trim(summary.get("conversationId")))
        candidates.extend(self._from_summary(summary))
        candidates.extend(self._from_profile(profile, source_id=_trim(summary.get("conversationId"))))
        if self.analyzer is not None:
            try:
                raw = self.analyzer.analyze_json(
                    system=SEMANTIC_EXTRACTION_SYSTEM_PROMPT,
                    user=build_semantic_extraction_prompt(
                        summary=_trim(summary.get("summary")),
                        existing_memories=existing or [],
                        learning={
                            "recurringMistakes": (learning or {}).get("recurringMistakes") or [],
                            "strengths": (learning or {}).get("strengths") or [],
                            "learningPatterns": (learning or {}).get("learningPatterns") or [],
                        },
                    ),
                )
                candidates.extend(self._from_llm(raw, source_id=_trim(summary.get("conversationId"))))
            except Exception as exc:  # noqa: BLE001
                call_log.warn("MEMORY", f"memory_extracted skip llm: {exc}")
        return self._validate_candidates(candidates)

    def extractAndIndexFromConversation(
        self,
        conversationId: str,
        *,
        userId: str | None = None,
    ) -> list[dict[str, Any]]:
        cid = _trim(conversationId)
        if not cid:
            raise ConversationNotFound()
        session = self.repo.find_conversation_session(cid)
        if session is None:
            raise ConversationNotFound()
        owner = _trim(userId)
        user_id = _trim(session.get("userId"))
        if owner and owner != user_id:
            raise ConversationAccessDenied()
        if session.get("status") != COMPLETED:
            raise ConversationNotCompleted()
        summary = self.repo.find_conversation_summary(cid)
        if summary is None or (
            summary.get("summaryStatus")
            and summary.get("summaryStatus") != COMPLETED
        ):
            raise SummaryNotFound()
        learning = None
        find_learning = getattr(self.repo, "find_learning_memory", None)
        if callable(find_learning):
            learning = find_learning(user_id)
        profile = None
        find_profile = getattr(self.repo, "find_user_profile", None)
        if callable(find_profile):
            profile = find_profile(user_id)
        existing = self.repo.list_memory_metadata(tenant_id=self.tenant_id, user_id=user_id)
        candidates = self.extractSemanticMemories(
            summary, learning=learning, existing=existing, profile=profile
        )
        stored: list[dict[str, Any]] = []
        for candidate in candidates:
            saved = self.upsertSemanticMemory(
                candidate,
                user_id=user_id,
                topic_id=session.get("topicId"),
                topic_level=_trim((self.repo.find_topic(session.get("topicId")) or {}).get("level")),
                source_conversation_id=cid,
            )
            if saved:
                stored.append(saved)
                call_log.info(
                    "MEMORY",
                    "memory_extracted",
                    extra={
                        "memoryId": saved.get("memoryId"),
                        "memoryType": saved.get("memoryType"),
                        "userId": user_id,
                    },
                )
        return stored

    def upsertSemanticMemory(
        self,
        candidate: dict[str, Any],
        *,
        user_id: str,
        topic_id: Any = None,
        topic_level: str = "",
        source_conversation_id: str = "",
    ) -> dict[str, Any] | None:
        tenant = self.tenant_id
        uid = _trim(user_id)
        if not tenant or not uid:
            return None
        identity = identity_key(
            candidate["memoryType"],
            _trim(candidate.get("skill")),
            candidate["content"],
        )
        point_id, memory_id = memory_ids(tenant_id=tenant, user_id=uid, identity=identity)
        existing = self.repo.find_memory_by_identity(
            tenant_id=tenant, user_id=uid, identity_key=identity
        )
        now = _utc_now()
        if existing and _trim(existing.get("lastIndexedSourceId")) == _trim(source_conversation_id):
            return existing
        duplicate = self._find_semantic_duplicate(candidate["content"], tenant_id=tenant, user_id=uid)
        if duplicate is not None:
            existing = existing or self.repo.find_memory_metadata(duplicate.get("memoryId") or duplicate.get("id"))
            self.metrics["memory_deduplicated"] += 1
            call_log.info(
                "MEMORY",
                "memory_deduplicated",
                extra={"memoryId": (existing or {}).get("memoryId"), "userId": uid},
            )
        frequency = 1
        first_seen = now
        confidence = float(candidate.get("confidence") or 0.7)
        importance = float(candidate.get("importance") or 0.5)
        if existing:
            frequency = int(existing.get("frequency") or 1) + 1
            first_seen = existing.get("createdAt") or now
            memory_id = _trim(existing.get("memoryId")) or memory_id
            point_id = _trim(existing.get("qdrantPointId")) or point_id
            confidence = max(confidence, float(existing.get("confidence") or 0))
            importance = max(importance, float(existing.get("importance") or 0))
        status = _trim(candidate.get("status")).upper() or "ACTIVE"
        if status not in STATUSES:
            status = "ACTIVE"
        scope = _trim(candidate.get("scope")).upper() or "GLOBAL"
        if scope not in SCOPES:
            scope = "GLOBAL"
        topic_value = str(topic_id) if scope == "TOPIC" and topic_id is not None else candidate.get("topicId")
        doc = {
            "tenantId": tenant,
            "userId": uid,
            "memoryId": memory_id,
            "identityKey": identity,
            "qdrantPointId": point_id,
            "memoryType": candidate["memoryType"],
            "category": candidate.get("category") or CATEGORY_FROM_TYPE.get(candidate["memoryType"]),
            "skill": _trim(candidate.get("skill")) or None,
            "content": candidate["content"],
            "sourceType": candidate.get("sourceType") or "conversation_summary",
            "sourceId": candidate.get("sourceId") or source_conversation_id,
            "topicId": str(topic_value) if topic_value else None,
            "topicLevel": _trim(topic_level) or _trim(candidate.get("topicLevel")) or None,
            "scope": scope,
            "status": status,
            "importance": importance,
            "confidence": confidence,
            "frequency": frequency,
            "createdAt": first_seen,
            "embeddingModel": self.embedding_model,
            "embeddingVersion": self.embedding_version,
            "indexed": False,
            "lastSeenAt": now,
            "lastIndexedSourceId": source_conversation_id or None,
            "updatedAt": now,
        }
        saved = self.repo.upsert_memory_metadata(doc)
        event = "memory_updated" if existing else "memory_created"
        self.metrics[event] += 1
        call_log.info(
            "MEMORY",
            event,
            extra={"memoryId": memory_id, "userId": uid},
        )
        self._index_record(saved)
        return saved

    def retrieveRelevantMemories(
        self,
        *,
        tenantId: str,
        userId: str,
        query: str,
        topicId: str | None = None,
        topicLevel: str | None = None,
        memoryTypes: list[str] | None = None,
        category: str | None = None,
        skill: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        context = RetrievalContext(
            tenant_id=_trim(tenantId) or self.tenant_id,
            user_id=_trim(userId),
            query=query,
            topic_id=_trim(topicId) or None,
            topic_level=_trim(topicLevel) or None,
            memory_types=tuple(item for item in (memoryTypes or []) if item) or None,
            category=_trim(category) or None,
            skill=_trim(skill) or None,
            status=_trim(status) or None,
            limit=limit,
        )
        return self.retrieval.retrieve(context)

    def retrieveForRuntime(
        self,
        *,
        user_id: str,
        topic_id: str = "",
        topic_title: str = "",
        topic_level: str = "",
        current_goal: str = "",
        goal_description: str = "",
        learning: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        return self.retrieval.retrieveForRuntime(
            tenant_id=_trim(tenant_id) or self.tenant_id,
            user_id=user_id,
            topic_id=topic_id,
            topic_title=topic_title,
            topic_level=topic_level,
            current_goal=current_goal,
            goal_description=goal_description,
            learning=learning,
            limit=limit,
        )

    def reindexUserMemories(self, userId: str, *, tenantId: str | None = None) -> int:
        tenant = _trim(tenantId) or self.tenant_id
        rows = self.repo.list_memory_metadata(tenant_id=tenant, user_id=_trim(userId))
        count = 0
        for row in rows:
            if self._index_record(row):
                count += 1
        return count

    def reindexTenantMemories(self, tenantId: str) -> int:
        tenant = _trim(tenantId) or self.tenant_id
        lister = getattr(self.repo, "list_memory_metadata_for_tenant", None)
        rows = lister(tenant) if callable(lister) else []
        count = 0
        for row in rows:
            if self._index_record(row):
                count += 1
        return count

    def deleteSemanticMemory(self, memoryId: str) -> None:
        row = self.repo.find_memory_metadata(memoryId)
        if not row:
            return
        try:
            self.vectors.deleteMemory(str(row.get("qdrantPointId") or memoryId))
        except Exception as exc:  # noqa: BLE001
            call_log.warn("MEMORY", f"memory_deleted qdrant skip: {exc}")
        self.repo.delete_memory_metadata(memoryId)
        call_log.info("MEMORY", "memory_deleted", extra={"memoryId": memoryId})

    def archiveSemanticMemory(self, memoryId: str) -> dict[str, Any] | None:
        row = self.repo.find_memory_metadata(memoryId)
        if not row:
            return None
        row["status"] = "ARCHIVED"
        row["updatedAt"] = _utc_now()
        saved = self.repo.upsert_memory_metadata(row)
        self._index_record(saved)
        call_log.info("MEMORY", "memory_archived", extra={"memoryId": memoryId})
        return saved

    def deleteUserSemanticMemories(self, tenantId: str, userId: str) -> None:
        tenant = _trim(tenantId) or self.tenant_id
        uid = _trim(userId)
        try:
            self.vectors.deleteUserMemories(tenant_id=tenant, user_id=uid)
        except Exception as exc:  # noqa: BLE001
            call_log.warn("MEMORY", f"memory_deleted qdrant skip: {exc}")
        self.repo.delete_user_memory_metadata(tenant_id=tenant, user_id=uid)
        call_log.info("MEMORY", "memory_deleted", extra={"tenantId": tenant, "userId": uid})

    def _index_record(self, row: dict[str, Any] | None) -> bool:
        if not row:
            return False
        try:
            embed_started = time.perf_counter()
            vector = self.embeddings.embed(_trim(row.get("content")))
            self.metrics["embedding_latency_ms"] = (time.perf_counter() - embed_started) * 1000
            payload = {
                "tenantId": row.get("tenantId"),
                "userId": row.get("userId"),
                "memoryId": row.get("memoryId"),
                "memoryType": row.get("memoryType"),
                "category": row.get("category"),
                "skill": row.get("skill"),
                "content": row.get("content"),
                "scope": row.get("scope") or "GLOBAL",
                "topicId": row.get("topicId"),
                "topicLevel": row.get("topicLevel"),
                "importance": float(row.get("importance") or 0),
                "confidence": float(row.get("confidence") or 0),
                "status": row.get("status") or "ACTIVE",
                "frequency": int(row.get("frequency") or 1),
                "sourceType": row.get("sourceType"),
                "sourceId": row.get("sourceId"),
                "embeddingModel": self.embedding_model,
                "embeddingVersion": self.embedding_version,
                "firstSeenAt": str(row.get("createdAt") or ""),
                "lastSeenAt": str(row.get("lastSeenAt") or row.get("updatedAt") or ""),
            }
            self.vectors.upsertMemory(
                point_id=str(row.get("qdrantPointId") or row.get("memoryId")),
                vector=vector,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            self.metrics["qdrant_failures"] += 1
            call_log.warn("MEMORY", f"memory_indexed skip: {exc}")
            return False
        row["indexed"] = True
        row["indexedAt"] = _utc_now()
        self.repo.upsert_memory_metadata(row)
        self.metrics["memory_indexed"] += 1
        call_log.info("MEMORY", "memory_indexed", extra={"memoryId": row.get("memoryId")})
        return True

    def _find_semantic_duplicate(
        self, content: str, *, tenant_id: str, user_id: str
    ) -> dict[str, Any] | None:
        try:
            vector = self.embeddings.embed(content)
            hits = self.vectors.searchMemories(
                vector=vector,
                tenant_id=tenant_id,
                user_id=user_id,
                limit=3,
            )
        except Exception:
            return None
        for hit in hits:
            if float(hit.get("score") or 0) >= self.config.similarity_threshold:
                payload = hit.get("payload") or {}
                payload["id"] = hit.get("id")
                return payload
        return None

    def _from_summary(self, summary: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not summary:
            return []
        source_id = _trim(summary.get("conversationId"))
        out: list[dict[str, Any]] = []
        for item in summary.get("importantFacts") or []:
            if isinstance(item, dict):
                text = _trim(item.get("fact"))
                confidence = _as_float(item.get("confidence")) or 0.9
            else:
                text = _trim(item)
                confidence = 0.9
            if not text:
                continue
            skill = ""
            lowered = text.lower()
            if "name" in lowered:
                skill = "name"
            elif any(token in lowered for token in ("live", "living", "from")):
                skill = "location"
            elif any(token in lowered for token in ("developer", "student", "teacher", "work")):
                skill = "profession"
            out.append(
                {
                    "memoryType": "PROFILE_FACT",
                    "category": "profile",
                    "skill": skill,
                    "content": text,
                    "importance": 0.82,
                    "confidence": confidence,
                    "status": "ACTIVE",
                    "scope": "GLOBAL",
                    "sourceType": "conversation_summary",
                    "sourceId": source_id,
                }
            )
        for item in summary.get("strengths") or []:
            text = _trim(item if not isinstance(item, dict) else item.get("description") or item.get("strength"))
            if not text:
                continue
            out.append(
                {
                    "memoryType": "LEARNING_STRENGTH",
                    "category": "learning",
                    "content": text,
                    "importance": 0.7,
                    "confidence": 0.84,
                    "status": "ACTIVE",
                    "scope": "GLOBAL",
                    "sourceType": "conversation_summary",
                    "sourceId": source_id,
                }
            )
        return out

    def _from_profile(
        self, profile: dict[str, Any] | None, *, source_id: str
    ) -> list[dict[str, Any]]:
        if not profile:
            return []
        blob = profile.get("profile") if isinstance(profile.get("profile"), dict) else profile
        out: list[dict[str, Any]] = []
        for key in ("profession", "education", "location", "experience", "nativeLanguage"):
            value = _trim((blob or {}).get(key))
            if not value:
                continue
            out.append(
                {
                    "memoryType": "PROFILE_FACT",
                    "category": "profile",
                    "skill": key,
                    "content": f"The learner's {key} is {value}.",
                    "importance": 0.8,
                    "confidence": 0.92,
                    "status": "ACTIVE",
                    "scope": "GLOBAL",
                    "sourceType": "profile_memory",
                    "sourceId": source_id,
                }
            )
        for field, skill in (("hobbies", "hobby"), ("interests", "interest")):
            for item in (blob or {}).get(field) or []:
                value = _trim(item)
                if not value:
                    continue
                out.append(
                    {
                        "memoryType": "PROFILE_FACT",
                        "category": "profile",
                        "skill": skill,
                        "content": f"The learner's {skill} is {value}.",
                        "importance": 0.76,
                        "confidence": 0.9,
                        "status": "ACTIVE",
                        "scope": "GLOBAL",
                        "sourceType": "profile_memory",
                        "sourceId": source_id,
                    }
                )
        return out

    def _from_learning(
        self, learning: dict[str, Any] | None, *, source_id: str
    ) -> list[dict[str, Any]]:
        if not learning:
            return []
        out: list[dict[str, Any]] = []
        for row in learning.get("recurringMistakes") or []:
            if not isinstance(row, dict):
                continue
            skill = _trim(row.get("skill"))
            issue = _trim(row.get("issue")) or f"User has a recurring difficulty with {skill}."
            status = _trim(row.get("status")).upper() or "ACTIVE"
            out.append(
                {
                    "memoryType": "LEARNING_WEAKNESS",
                    "category": "grammar" if _trim(row.get("category")) == "grammar" else _trim(row.get("category")) or "learning",
                    "skill": skill,
                    "content": issue,
                    "importance": min(1.0, 0.45 + 0.1 * int(row.get("frequency") or 1)),
                    "confidence": float(row.get("confidence") or 0.86),
                    "status": status if status in STATUSES else "ACTIVE",
                    "scope": "GLOBAL",
                    "sourceType": "learning_memory",
                    "sourceId": source_id,
                }
            )
        for row in learning.get("strengths") or []:
            if not isinstance(row, dict):
                continue
            text = _trim(row.get("description") or row.get("strength"))
            if not text:
                continue
            out.append(
                {
                    "memoryType": "LEARNING_STRENGTH",
                    "category": "learning",
                    "skill": _trim(row.get("skill")),
                    "content": text,
                    "importance": 0.7,
                    "confidence": float(row.get("confidence") or 0.84),
                    "status": "ACTIVE",
                    "scope": "GLOBAL",
                    "sourceType": "learning_memory",
                    "sourceId": source_id,
                }
            )
        for row in learning.get("learningPatterns") or []:
            if not isinstance(row, dict):
                continue
            text = _trim(row.get("pattern") or row.get("description"))
            if not text:
                continue
            out.append(
                {
                    "memoryType": "LEARNING_PATTERN",
                    "category": "learning",
                    "content": text,
                    "importance": 0.72,
                    "confidence": float(row.get("confidence") or 0.84),
                    "status": "ACTIVE",
                    "scope": "GLOBAL",
                    "sourceType": "learning_memory",
                    "sourceId": source_id,
                }
            )
        return out

    def _from_llm(self, raw: Any, *, source_id: str) -> list[dict[str, Any]]:
        if isinstance(raw, str):
            try:
                raw = parse_json_object(raw)
            except (ValueError, TypeError):
                return []
        if not isinstance(raw, dict):
            return []
        out: list[dict[str, Any]] = []
        for item in raw.get("memories") or []:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item["sourceType"] = "conversation_summary"
            item["sourceId"] = source_id
            out.append(item)
        return out

    def _validate_candidates(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in candidates:
            memory_type = _trim(item.get("memoryType")).upper()
            if memory_type not in MEMORY_TYPES:
                continue
            content = _trim(item.get("content"))
            if len(content) < self.config.min_content_length:
                continue
            if _POLLUTION.search(content):
                continue
            if memory_type == "PROFILE_FACT" and _norm(item.get("skill")) in SKIP_PROFILE_KEYS:
                continue
            confidence = _as_float(item.get("confidence"))
            importance = _as_float(item.get("importance"))
            if confidence is None or confidence < self.config.min_confidence:
                continue
            if importance is None or importance < self.config.min_importance:
                continue
            if not _trim(item.get("sourceId")):
                continue
            key = identity_key(memory_type, _trim(item.get("skill")), content)
            if key in seen:
                continue
            seen.add(key)
            category = _trim(item.get("category")) or CATEGORY_FROM_TYPE.get(memory_type, "conversation")
            item["memoryType"] = memory_type
            item["category"] = category
            item["content"] = content
            item["confidence"] = confidence
            item["importance"] = importance
            out.append(item)
            if len(out) >= self.config.max_candidates:
                break
        return out
