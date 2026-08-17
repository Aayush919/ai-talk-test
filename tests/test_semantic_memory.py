"""Qdrant semantic memory — retrieval layer, tenant-safe, Mongo remains source of truth."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.runtime.service import ConversationRuntimeService
from core.semantic.config import SemanticMemoryConfig
from core.semantic.embeddings import HashingEmbeddingProvider, cosine_similarity
from core.semantic.repository import InMemorySemanticRepository
from core.semantic.service import (
    SemanticMemoryService,
    build_retrieval_query,
    compact_memory_context,
    identity_key,
    memory_ids,
)
from wrappers.qdrant_store import QdrantStore


def _now():
    return datetime.now(timezone.utc)


def _candidate(**overrides) -> dict:
    row = {
        "memoryType": "LEARNING_WEAKNESS",
        "category": "grammar",
        "skill": "past_tense",
        "content": "User frequently struggles with past tense when describing previous experiences.",
        "importance": 0.82,
        "confidence": 0.91,
        "scope": "GLOBAL",
        "sourceType": "conversation_summary",
        "sourceId": "sum_1",
        "status": "ACTIVE",
    }
    row.update(overrides)
    return row


class FakeAnalyzer:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload if payload is not None else {"memories": []}
        self.calls = 0

    def analyze_json(self, *, system: str, user: str) -> dict:
        self.calls += 1
        return self.payload


class FakeRepo:
    def __init__(self) -> None:
        self.sessions: list[dict] = []
        self.summaries: list[dict] = []
        self.topics: list[dict] = [
            {
                "_id": "topic_intro",
                "title": "Introduction",
                "level": "A1",
                "goals": [
                    {"key": "introduce_self", "description": "User can introduce their name."},
                    {"key": "talk_about_work", "description": "User can talk about work."},
                ],
            }
        ]
        self.progress: list[dict] = []
        self.profiles: list[dict] = []
        self.learning: list[dict] = []
        self.memory: list[dict] = []

    def find_conversation_session(self, conversation_id: str) -> dict | None:
        for row in self.sessions:
            if str(row["_id"]) == str(conversation_id):
                return dict(row)
        return None

    def find_conversation_summary(self, conversation_id: str) -> dict | None:
        for row in self.summaries:
            if str(row["conversationId"]) == str(conversation_id):
                return dict(row)
        return None

    def find_topic(self, topic_id) -> dict | None:
        for topic in self.topics:
            if topic["_id"] == topic_id or str(topic["_id"]) == str(topic_id):
                return dict(topic)
        return None

    def find_progress(self, user_id: str, topic_id) -> dict | None:
        for row in self.progress:
            if row["userId"] == user_id and str(row["topicId"]) == str(topic_id):
                return dict(row)
        return None

    def find_user_profile(self, user_id: str) -> dict | None:
        for row in self.profiles:
            if str(row["userId"]) == str(user_id):
                return dict(row)
        return None

    def find_learning_memory(self, user_id: str) -> dict | None:
        for row in self.learning:
            if str(row["userId"]) == str(user_id):
                return dict(row)
        return None

    def find_memory_metadata(self, memory_id: str) -> dict | None:
        for row in self.memory:
            if str(row.get("memoryId")) == str(memory_id):
                return dict(row)
        return None

    def find_memory_by_identity(self, *, tenant_id: str, user_id: str, identity_key: str) -> dict | None:
        for row in self.memory:
            if (
                str(row.get("tenantId")) == str(tenant_id)
                and str(row.get("userId")) == str(user_id)
                and str(row.get("identityKey")) == str(identity_key)
            ):
                return dict(row)
        return None

    def list_memory_metadata(self, *, tenant_id: str, user_id: str, indexed_only: bool | None = None) -> list[dict]:
        out = []
        for row in self.memory:
            if str(row.get("tenantId")) != str(tenant_id) or str(row.get("userId")) != str(user_id):
                continue
            if indexed_only is True and not row.get("indexed"):
                continue
            if indexed_only is False and row.get("indexed"):
                continue
            out.append(dict(row))
        return out

    def list_memory_metadata_for_tenant(self, tenant_id: str) -> list[dict]:
        return [dict(row) for row in self.memory if str(row.get("tenantId")) == str(tenant_id)]

    def upsert_memory_metadata(self, doc: dict) -> dict:
        payload = dict(doc)
        memory_id = str(payload.get("memoryId") or "")
        for row in self.memory:
            if str(row.get("memoryId")) == memory_id:
                created = row.get("createdAt")
                row.update(payload)
                if created and "createdAt" not in doc:
                    row["createdAt"] = created
                return dict(row)
        payload.setdefault("createdAt", _now())
        self.memory.append(payload)
        return dict(payload)

    def delete_memory_metadata(self, memory_id: str) -> None:
        self.memory = [row for row in self.memory if str(row.get("memoryId")) != str(memory_id)]

    def delete_user_memory_metadata(self, *, tenant_id: str, user_id: str) -> None:
        self.memory = [
            row
            for row in self.memory
            if not (
                str(row.get("tenantId")) == str(tenant_id)
                and str(row.get("userId")) == str(user_id)
            )
        ]


def _service(
    repo: FakeRepo | None = None,
    *,
    tenant_id: str = "talkengly",
    vectors: InMemorySemanticRepository | None = None,
    analyzer: FakeAnalyzer | None = None,
    config: dict | None = None,
) -> SemanticMemoryService:
    return SemanticMemoryService(
        repo or FakeRepo(),
        vectors=vectors if vectors is not None else InMemorySemanticRepository(),
        embeddings=HashingEmbeddingProvider(dimension=64, model="hashing-v1", version="v1"),
        analyzer=analyzer,
        config=config,
        tenant_id=tenant_id,
        embedding_model="hashing-v1",
        embedding_version="v1",
    )


def _completed(repo: FakeRepo, *, user_id: str = "USER_A", cid: str | None = None) -> str:
    conversation_id = cid or str(uuid4())
    repo.sessions.append(
        {
            "_id": conversation_id,
            "userId": user_id,
            "topicId": "topic_intro",
            "status": "COMPLETED",
        }
    )
    repo.summaries.append(
        {
            "conversationId": conversation_id,
            "userId": user_id,
            "summaryStatus": "COMPLETED",
            "summary": (
                "User talked about a React project confidently. "
                "User struggled when describing previous work experience."
            ),
        }
    )
    repo.progress.append(
        {
            "userId": user_id,
            "topicId": "topic_intro",
            "status": "IN_PROGRESS",
            "progress": 20,
            "goalsCompleted": [],
            "goalsRemaining": ["introduce_self", "talk_about_work"],
        }
    )
    return conversation_id


def test_tenant_isolation_never_returns_other_tenant_memory():
    vectors = InMemorySemanticRepository()
    repo_a = FakeRepo()
    repo_b = FakeRepo()
    svc_a = _service(repo_a, tenant_id="tenant_a", vectors=vectors)
    svc_b = _service(repo_b, tenant_id="tenant_b", vectors=vectors)
    svc_a.upsertSemanticMemory(_candidate(sourceId="a1"), user_id="USER_A")
    svc_b.upsertSemanticMemory(
        _candidate(
            skill="articles",
            content="User frequently omits articles when describing workplace meetings.",
            sourceId="b1",
        ),
        user_id="USER_B",
    )
    hits = svc_a.retrieveRelevantMemories(
        tenantId="tenant_a",
        userId="USER_A",
        query="User frequently struggles with past tense when describing previous experiences.",
    )
    contents = [row["content"] for row in hits]
    assert any("past tense" in text for text in contents)
    assert all("articles" not in text for text in contents)
    assert svc_a.retrieveRelevantMemories(
        tenantId="tenant_a",
        userId="USER_B",
        query="User frequently omits articles when describing workplace meetings.",
    ) == []


def test_same_user_id_different_tenant_stays_isolated():
    vectors = InMemorySemanticRepository()
    repo_a = FakeRepo()
    repo_b = FakeRepo()
    svc_a = _service(repo_a, tenant_id="tenant_a", vectors=vectors)
    svc_b = _service(repo_b, tenant_id="tenant_b", vectors=vectors)
    svc_a.upsertSemanticMemory(_candidate(sourceId="a1"), user_id="SHARED")
    svc_b.upsertSemanticMemory(
        _candidate(
            content="User prefers cricket and talks about matches with confidence.",
            memoryType="PREFERENCE",
            skill="",
            sourceId="b1",
        ),
        user_id="SHARED",
    )
    hits = svc_a.retrieveRelevantMemories(
        tenantId="tenant_a",
        userId="SHARED",
        query="User frequently struggles with past tense when describing previous experiences.",
    )
    assert hits
    assert all("cricket" not in (row.get("content") or "") for row in hits)
    ids_a = {row.get("memoryId") for row in hits}
    hits_b = svc_b.retrieveRelevantMemories(
        tenantId="tenant_b",
        userId="SHARED",
        query="cricket matches",
    )
    assert {row.get("memoryId") for row in hits_b}.isdisjoint(ids_a)


def test_trivial_and_low_quality_memories_are_rejected():
    svc = _service(analyzer=FakeAnalyzer({
        "memories": [
            _candidate(content="User said hello.", confidence=0.99, importance=0.99),
            _candidate(content="User said yes.", skill="fluency", confidence=0.99, importance=0.99),
            _candidate(
                content="User paused for 1 second during the greeting.",
                skill="pause",
                confidence=0.99,
                importance=0.99,
            ),
            _candidate(content="User frequently struggles with past tense.", confidence=0.4),
            _candidate(
                content="User is comfortable discussing software development.",
                memoryType="LEARNING_STRENGTH",
                skill="fluency",
                importance=0.2,
            ),
            _candidate(
                content="User is comfortable discussing software development.",
                memoryType="LEARNING_STRENGTH",
                skill="fluency",
                importance=0.2,
            ),
        ]
    }))
    kept = svc.extractSemanticMemories({"conversationId": "c1", "summary": "hello"})
    assert kept == []


def test_missing_source_id_is_rejected():
    svc = _service()
    kept = svc._validate_candidates([_candidate(sourceId="")])
    assert kept == []


def test_identity_dedup_updates_frequency_instead_of_duplicating():
    repo = FakeRepo()
    svc = _service(repo)
    first = svc.upsertSemanticMemory(_candidate(sourceId="c1"), user_id="USER_A", source_conversation_id="c1")
    second = svc.upsertSemanticMemory(
        _candidate(
            content="User has difficulty using past tense correctly in work stories.",
            sourceId="c2",
        ),
        user_id="USER_A",
        source_conversation_id="c2",
    )
    assert first["memoryId"] == second["memoryId"]
    assert second["frequency"] == 2
    assert len(repo.memory) == 1
    assert len(svc.vectors.points) == 1


def test_semantic_dedup_merges_similar_memories():
    repo = FakeRepo()
    svc = _service(repo, config={"similarityThreshold": 0.55})
    first = svc.upsertSemanticMemory(
        _candidate(skill="", sourceId="c1"),
        user_id="USER_A",
        source_conversation_id="c1",
    )
    second = svc.upsertSemanticMemory(
        _candidate(
            skill="",
            content="User frequently struggles with past tense when talking about previous experiences.",
            sourceId="c2",
        ),
        user_id="USER_A",
        source_conversation_id="c2",
    )
    assert first["memoryId"] == second["memoryId"]
    assert second["frequency"] == 2
    assert svc.metrics["memory_deduplicated"] >= 1


def test_mongodb_stays_correct_when_qdrant_is_offline():
    repo = FakeRepo()
    vectors = InMemorySemanticRepository()
    vectors.online = False
    svc = _service(repo, vectors=vectors)
    saved = svc.upsertSemanticMemory(_candidate(), user_id="USER_A", source_conversation_id="c1")
    assert saved is not None
    assert saved["indexed"] is False
    assert repo.find_memory_metadata(saved["memoryId"])["content"]
    assert svc.vectors.points == {}
    assert svc.qdrantHealthCheck() == "QDRANT_UNAVAILABLE"


def test_deterministic_ids_are_stable():
    left = memory_ids(tenant_id="talkengly", user_id="USER_A", identity="LEARNING_WEAKNESS|past_tense")
    right = memory_ids(tenant_id="talkengly", user_id="USER_A", identity="LEARNING_WEAKNESS|past_tense")
    other = memory_ids(tenant_id="other", user_id="USER_A", identity="LEARNING_WEAKNESS|past_tense")
    assert left == right
    assert left != other
    assert identity_key("LEARNING_WEAKNESS", "past_tense", "anything") == "LEARNING_WEAKNESS|past_tense"


def test_qdrant_payload_is_not_a_transcript():
    svc = _service()
    saved = svc.upsertSemanticMemory(_candidate(), user_id="USER_A")
    point = svc.vectors.getMemoryById(saved["qdrantPointId"])
    payload = point["payload"]
    assert "transcript" not in payload
    assert "messages" not in payload
    assert payload["tenantId"] == "talkengly"
    assert payload["userId"] == "USER_A"
    assert payload["embeddingVersion"] == "v1"
    assert "User frequently struggles" in payload["content"]
    assert "React project" not in payload["content"]


def test_delete_user_memories_is_tenant_safe():
    vectors = InMemorySemanticRepository()
    repo = FakeRepo()
    svc_a = _service(repo, tenant_id="tenant_a", vectors=vectors)
    svc_b = _service(repo, tenant_id="tenant_b", vectors=vectors)
    a = svc_a.upsertSemanticMemory(_candidate(sourceId="a"), user_id="SHARED")
    b = svc_b.upsertSemanticMemory(
        _candidate(skill="articles", content="User frequently omits articles in introductions.", sourceId="b"),
        user_id="SHARED",
    )
    svc_a.deleteUserSemanticMemories("tenant_a", "SHARED")
    assert repo.find_memory_metadata(a["memoryId"]) is None
    assert repo.find_memory_metadata(b["memoryId"]) is not None
    assert vectors.getMemoryById(b["qdrantPointId"]) is not None


def test_resolved_memories_do_not_dominate_retrieval():
    svc = _service()
    svc.upsertSemanticMemory(
        _candidate(status="RESOLVED", sourceId="old"),
        user_id="USER_A",
        source_conversation_id="old",
    )
    svc.upsertSemanticMemory(
        _candidate(
            memoryType="LEARNING_WEAKNESS",
            skill="vocabulary",
            content="User often searches for workplace vocabulary while speaking.",
            status="ACTIVE",
            sourceId="new",
        ),
        user_id="USER_A",
        source_conversation_id="new",
    )
    hits = svc.retrieveRelevantMemories(
        tenantId="talkengly",
        userId="USER_A",
        query="workplace vocabulary while speaking English",
    )
    assert hits
    assert hits[0]["skill"] == "vocabulary"


def test_extract_and_index_from_learning_memory():
    repo = FakeRepo()
    cid = _completed(repo)
    repo.learning.append(
        {
            "userId": "USER_A",
            "recurringMistakes": [
                {
                    "skill": "past_tense",
                    "category": "grammar",
                    "issue": "User frequently struggles with past tense when describing previous experiences.",
                    "frequency": 5,
                    "confidence": 0.91,
                    "status": "ACTIVE",
                }
            ],
            "learningPatterns": [
                {
                    "pattern": "User gives longer answers when the AI asks specific follow-up questions.",
                    "confidence": 0.84,
                }
            ],
            "strengths": [],
        }
    )
    svc = _service(repo)
    stored = svc.extractAndIndexFromConversation(cid, userId="USER_A")
    assert stored
    types = {row["memoryType"] for row in stored}
    assert "LEARNING_WEAKNESS" in types
    assert "LEARNING_PATTERN" in types
    assert all(row.get("indexed") for row in stored)


def test_summary_facts_are_indexed_without_llm():
    repo = FakeRepo()
    cid = _completed(repo)
    repo.summaries[0]["importantFacts"] = [
        {"fact": "The learner is a software developer.", "confidence": 0.95},
        {"fact": "The learner lives in Chandigarh.", "confidence": 0.93},
    ]
    svc = _service(repo)
    stored = svc.extractAndIndexFromConversation(cid, userId="USER_A")
    contents = " ".join(row.get("content") or "" for row in stored).lower()
    assert "software developer" in contents
    assert "chandigarh" in contents
    assert repo.memory


def test_same_conversation_is_idempotent():
    repo = FakeRepo()
    cid = _completed(repo)
    repo.learning.append(
        {
            "userId": "USER_A",
            "recurringMistakes": [
                {
                    "skill": "past_tense",
                    "issue": "User frequently struggles with past tense when describing previous experiences.",
                    "frequency": 2,
                    "confidence": 0.9,
                    "status": "ACTIVE",
                }
            ],
        }
    )
    svc = _service(repo)
    first = svc.extractAndIndexFromConversation(cid, userId="USER_A")
    second = svc.extractAndIndexFromConversation(cid, userId="USER_A")
    assert len(first) == 1
    assert second[0]["frequency"] == first[0]["frequency"]


def test_reindex_and_archive():
    repo = FakeRepo()
    svc = _service(repo)
    saved = svc.upsertSemanticMemory(_candidate(), user_id="USER_A")
    svc.vectors.points.clear()
    repo.memory[0]["indexed"] = False
    count = svc.reindexUserMemories("USER_A")
    assert count == 1
    archived = svc.archiveSemanticMemory(saved["memoryId"])
    assert archived["status"] == "ARCHIVED"
    payload = svc.vectors.getMemoryById(saved["qdrantPointId"])["payload"]
    assert payload["status"] == "ARCHIVED"


def test_offline_qdrant_store_does_not_search():
    store = QdrantStore("")
    assert store.healthCheck() == "QDRANT_UNAVAILABLE"
    with pytest.raises(RuntimeError, match="QDRANT_UNAVAILABLE"):
        store.searchMemories(vector=[0.1, 0.2], tenant_id="talkengly", user_id="USER_A")


def test_retrieval_query_is_more_than_topic_title():
    query = build_retrieval_query(
        topic_title="Daily Routine",
        current_goal="talk_about_yesterday",
        learning_focus="past_tense",
    )
    assert "Daily Routine" in query
    assert "past tense" in query
    assert query != "Daily Routine"


def test_compact_memory_context_is_short():
    lines = compact_memory_context(
        [
            {
                "content": "User frequently struggles with past tense.",
                "memoryType": "LEARNING_WEAKNESS",
            }
        ]
    )
    assert lines == ["User frequently struggles with past tense"]


def test_runtime_caches_memories_and_survives_qdrant_failure():
    repo = FakeRepo()
    user_id = "USER_A"
    cid = str(uuid4())
    repo.sessions.append(
        {
            "_id": cid,
            "userId": user_id,
            "topicId": "topic_intro",
            "status": "ACTIVE",
        }
    )
    repo.progress.append(
        {
            "userId": user_id,
            "topicId": "topic_intro",
            "status": "IN_PROGRESS",
            "progress": 0,
            "goalsCompleted": [],
            "goalsRemaining": ["introduce_self", "talk_about_work"],
        }
    )

    class CountingSemantic:
        def __init__(self) -> None:
            self.calls = 0
            self.config = SemanticMemoryConfig.from_mapping({"retrieveOnGoalChange": False})

        def retrieveRelevantMemories(self, **kwargs):
            self.calls += 1
            assert kwargs["tenantId"] == "talkengly"
            assert kwargs["userId"] == user_id
            return [
                {
                    "memoryId": "MEM_123",
                    "type": "LEARNING_WEAKNESS",
                    "memoryType": "LEARNING_WEAKNESS",
                    "content": "User frequently struggles with past tense.",
                    "relevanceScore": 0.88,
                }
            ]

    semantic = CountingSemantic()
    runtime = ConversationRuntimeService(repo, semantic=semantic, tenant_id="talkengly")
    state = runtime.initializeConversationRuntime(cid)
    assert state["relevantMemories"][0]["content"].startswith("User frequently struggles")
    runtime.handleUserTurn(cid, "I work as a software engineer in Pune.")
    runtime.previewResponse(cid, "I like cricket too.")
    assert semantic.calls == 1

    class FailingSemantic:
        config = SemanticMemoryConfig.from_mapping()

        def retrieveRelevantMemories(self, **kwargs):
            raise RuntimeError("QDRANT_UNAVAILABLE")

    cid2 = str(uuid4())
    repo.sessions.append(
        {
            "_id": cid2,
            "userId": user_id,
            "topicId": "topic_intro",
            "status": "ACTIVE",
        }
    )
    runtime2 = ConversationRuntimeService(repo, semantic=FailingSemantic(), tenant_id="talkengly")
    failed = runtime2.initializeConversationRuntime(cid2)
    assert failed["userId"] == user_id
    assert failed["relevantMemories"] == []


def test_hashing_embeddings_are_deterministic_and_sized():
    embedder = HashingEmbeddingProvider(dimension=32)
    left = embedder.embed("User struggles with past tense.")
    right = embedder.embed("User struggles with past tense.")
    other = embedder.embed("User likes cricket.")
    assert len(left) == 32
    assert left == right
    assert cosine_similarity(left, right) > cosine_similarity(left, other)
