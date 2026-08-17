"""Step 11 — retrieve relevant long-term memories into LangGraph runtime."""

from __future__ import annotations

from uuid import uuid4

from core.runtime.prompts import build_generate_user_prompt
from core.runtime.service import ConversationRuntimeService
from core.semantic.config import SemanticMemoryConfig
from core.semantic.embeddings import HashingEmbeddingProvider
from core.semantic.repository import InMemorySemanticRepository
from core.semantic.retrieval import (
    build_retrieval_context,
    compact_memory_context,
)
from core.semantic.service import SemanticMemoryService
from tests.test_semantic_memory import FakeRepo, _candidate, _completed, _service


def test_retrieval_query_uses_topic_goal_and_learning_focus():
    context = build_retrieval_context(
        tenant_id="talkengly",
        user_id="USER_A",
        topic_id="daily_routine",
        topic_title="Daily Routine",
        current_goal="talk_about_yesterday",
        goal_description="Talk about yesterday",
        learning={
            "recurringMistakes": [
                {"skill": "past_tense", "status": "ACTIVE"},
            ]
        },
    )
    assert "Daily Routine" in context.query
    assert "yesterday" in context.query
    assert "past tense" in context.query
    assert context.query != "Daily Routine"
    assert "talkengly" in context.fingerprint
    assert "USER_A" in context.fingerprint


def test_mongo_enrichment_overrides_stale_qdrant_payload():
    repo = FakeRepo()
    svc = _service(repo)
    saved = svc.upsertSemanticMemory(_candidate(sourceId="c1"), user_id="USER_A")
    point = svc.vectors.points[saved["qdrantPointId"]]
    point["payload"]["content"] = "stale qdrant text"
    point["payload"]["frequency"] = 1
    repo.memory[0]["content"] = (
        "User frequently struggles with past tense when describing previous experiences."
    )
    repo.memory[0]["frequency"] = 5
    hits = svc.retrieveRelevantMemories(
        tenantId="talkengly",
        userId="USER_A",
        query=saved["content"],
    )
    assert hits
    assert "stale qdrant text" not in hits[0]["content"]
    assert "past tense" in hits[0]["content"]


def test_archived_mongo_memory_is_not_returned():
    repo = FakeRepo()
    svc = _service(repo)
    saved = svc.upsertSemanticMemory(_candidate(sourceId="c1"), user_id="USER_A")
    svc.archiveSemanticMemory(saved["memoryId"])
    hits = svc.retrieveRelevantMemories(
        tenantId="talkengly",
        userId="USER_A",
        query=saved["content"],
    )
    assert hits == []


def test_hashing_keeps_qdrant_hits_for_topic_query():
    """Local hashing scores ~0.2, below the dense 0.55 floor. Qdrant must still win."""
    repo = FakeRepo()
    svc = _service(repo)
    saved = svc.upsertSemanticMemory(_candidate(sourceId="c1"), user_id="USER_A")
    hits, _key = svc.retrieveForRuntime(
        user_id="USER_A",
        topic_title="Daily Routine",
        current_goal="talk_about_yesterday",
        goal_description="Talk about yesterday",
        tenant_id="talkengly",
    )
    assert hits
    assert saved["memoryId"] in {row["memoryId"] for row in hits}
    assert "past tense" in hits[0]["content"]


def test_qdrant_outage_falls_back_to_mongo_memories():
    repo = FakeRepo()
    svc = _service(repo)
    saved = svc.upsertSemanticMemory(_candidate(sourceId="c1"), user_id="USER_A")
    svc.vectors.online = False
    hits = svc.retrieveRelevantMemories(
        tenantId="talkengly",
        userId="USER_A",
        query="past tense previous experiences",
    )
    assert hits
    assert hits[0]["memoryId"] == saved["memoryId"]
    assert "past tense" in hits[0]["content"]


def test_stale_qdrant_point_without_mongo_is_dropped():
    repo = FakeRepo()
    vectors = InMemorySemanticRepository()
    embedder = HashingEmbeddingProvider(dimension=64)
    svc = SemanticMemoryService(
        repo,
        vectors=vectors,
        embeddings=embedder,
        tenant_id="talkengly",
    )
    vectors.upsertMemory(
        point_id="orphan",
        vector=embedder.embed("User frequently struggles with past tense."),
        payload={
            "tenantId": "talkengly",
            "userId": "USER_A",
            "memoryId": "missing",
            "memoryType": "LEARNING_WEAKNESS",
            "content": "User frequently struggles with past tense.",
            "status": "ACTIVE",
            "importance": 0.9,
            "confidence": 0.9,
            "scope": "GLOBAL",
        },
    )
    hits = svc.retrieveRelevantMemories(
        tenantId="talkengly",
        userId="USER_A",
        query="User frequently struggles with past tense.",
    )
    assert hits == []


def test_runtime_injects_compact_context_and_does_not_query_every_turn():
    repo = FakeRepo()
    cid = _completed(repo, user_id="USER_A")
    repo.sessions[0]["status"] = "ACTIVE"
    svc = _service(repo)
    svc.upsertSemanticMemory(_candidate(sourceId="c1"), user_id="USER_A")

    class CountingSemantic(SemanticMemoryService):
        def __init__(self) -> None:
            super().__init__(
                repo,
                vectors=svc.vectors,
                embeddings=svc.embeddings,
                config=SemanticMemoryConfig.from_mapping({"retrieveEveryTurn": False}),
                tenant_id="talkengly",
            )
            self.calls = 0

        def retrieveForRuntime(self, **kwargs):
            self.calls += 1
            return super().retrieveForRuntime(**kwargs)

    semantic = CountingSemantic()
    runtime = ConversationRuntimeService(repo, semantic=semantic, tenant_id="talkengly")
    state = runtime.initializeConversationRuntime(cid)
    assert state["relevantMemories"]
    assert "past tense" in state["relevantMemories"][0]["content"]
    prompt = build_generate_user_prompt(state, "I went to work yesterday.")
    assert "Relevant learner context" in prompt
    assert "relevanceScore" not in prompt
    assert "qdrant" not in prompt.lower()
    assert "goalEvidence" not in prompt
    runtime.previewResponse(cid, "I went to work yesterday.")
    runtime.applyCommittedTurn(
        cid,
        userText="I went to work yesterday.",
        assistantText="Nice. What did you do after work?",
    )
    assert semantic.calls == 1


def test_goal_change_refreshes_memories_once():
    repo = FakeRepo()
    cid = _completed(repo, user_id="USER_A")
    repo.sessions[0]["status"] = "ACTIVE"
    svc = _service(repo)
    svc.upsertSemanticMemory(_candidate(sourceId="c1"), user_id="USER_A")

    class CountingSemantic(SemanticMemoryService):
        def __init__(self) -> None:
            super().__init__(
                repo,
                vectors=svc.vectors,
                embeddings=svc.embeddings,
                tenant_id="talkengly",
            )
            self.calls = 0

        def retrieveForRuntime(self, **kwargs):
            self.calls += 1
            return super().retrieveForRuntime(**kwargs)

    semantic = CountingSemantic()
    runtime = ConversationRuntimeService(repo, semantic=semantic, tenant_id="talkengly")
    runtime.initializeConversationRuntime(cid)
    runtime.applyCommittedTurn(
        cid,
        userText="I am a software engineer.",
        assistantText="What do you do at work?",
        targetGoalId="talk_about_work",
    )
    assert semantic.calls == 2


def test_compact_context_hides_retrieval_metadata():
    lines = compact_memory_context(
        [
            {
                "memoryId": "MEM_123",
                "type": "LEARNING_WEAKNESS",
                "content": "User frequently struggles with past tense.",
                "relevanceScore": 0.88,
                "score": 0.91,
            }
        ]
    )
    assert lines == ["User frequently struggles with past tense"]


def test_tenant_filter_still_required_during_retrieval():
    vectors = InMemorySemanticRepository()
    svc_a = _service(FakeRepo(), tenant_id="tenant_a", vectors=vectors)
    svc_b = _service(FakeRepo(), tenant_id="tenant_b", vectors=vectors)
    svc_a.upsertSemanticMemory(_candidate(sourceId="a"), user_id="SHARED")
    svc_b.upsertSemanticMemory(
        _candidate(
            skill="articles",
            content="User frequently omits articles when describing workplace meetings.",
            sourceId="b",
        ),
        user_id="SHARED",
    )
    hits = svc_a.retrieveForRuntime(
        user_id="SHARED",
        topic_title="Work",
        current_goal="talk_about_work",
        tenant_id="tenant_a",
    )[0]
    assert hits
    assert all("articles" not in (row.get("content") or "") for row in hits)
