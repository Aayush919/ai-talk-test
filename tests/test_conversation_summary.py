"""Conversation summary — COMPLETED calls only, structured analysis, idempotent."""

from __future__ import annotations

import threading
from uuid import uuid4

import pytest

from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationNotCompleted,
    InsufficientConversationData,
    SummaryGenerationFailed,
)
from core.conversations.summary_service import ConversationSummaryService


def _topic() -> dict:
    return {
        "_id": "t1",
        "title": "Introduction",
        "description": "Help the learner introduce themselves in simple English.",
        "slug": "a1-introduction",
        "level": "A1",
        "goals": [
            {"key": "name", "description": "User can introduce their name."},
            {"key": "location", "description": "User can talk about where they live."},
            {"key": "hobbies", "description": "User can talk about basic hobbies."},
        ],
    }


def _analysis(**overrides) -> dict:
    payload = {
        "summary": (
            "Aayush introduced himself and talked about his work as a software "
            "developer. He described his profession clearly but made a grammar mistake."
        ),
        "keyPoints": [
            "User is a software developer.",
            "User works mainly with web applications.",
        ],
        "goals": [
            {
                "goalId": "name",
                "status": "COMPLETED",
                "evidence": "User introduced himself as Aayush.",
            },
            {
                "goalId": "location",
                "status": "PARTIAL",
                "evidence": "User mentioned work but not a home city.",
            },
            {
                "goalId": "hobbies",
                "status": "NOT_ATTEMPTED",
                "evidence": "",
            },
        ],
        "mistakes": [
            {
                "type": "GRAMMAR",
                "userText": "I am working here since two years.",
                "correction": "I have been working here for two years.",
                "explanation": "Use present perfect continuous with 'for'.",
            }
        ],
        "corrections": [
            {
                "original": "I am working here since two years.",
                "corrected": "I have been working here for two years.",
                "category": "GRAMMAR",
            }
        ],
        "strengths": ["User answers direct questions clearly."],
        "weaknesses": ["Present perfect with since/for."],
        "importantFacts": [
            {"fact": "User name is Aayush.", "confidence": 0.99},
            {"fact": "User is a software developer.", "confidence": 0.98},
        ],
        "vocabulary": [
            {
                "word": "profession",
                "meaning": "a person's occupation",
                "context": "Talking about work",
            }
        ],
        "grammarPatterns": [
            "Incorrect use of present perfect with 'since' and 'for'"
        ],
        "fluencyObservations": [
            "User responds confidently to direct questions."
        ],
    }
    payload.update(overrides)
    return payload


class FakeAnalyzer:
    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.payload = payload if payload is not None else _analysis()
        self.error = error
        self.calls = 0

    def analyze_json(self, *, system: str, user: str) -> dict:
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload


class FakeRepo:
    def __init__(self) -> None:
        self.sessions: list[dict] = []
        self.messages: list[dict] = []
        self.topics: list[dict] = [_topic()]
        self.summaries: list[dict] = []
        self._lock = threading.Lock()

    def find_conversation_session(self, conversation_id: str) -> dict | None:
        for row in self.sessions:
            if str(row["_id"]) == str(conversation_id):
                return dict(row)
        return None

    def find_topic(self, topic_id) -> dict | None:
        for topic in self.topics:
            if topic["_id"] == topic_id or str(topic["_id"]) == str(topic_id):
                return topic
        return None

    def list_messages(self, conversation_id: str) -> list[dict]:
        rows = [
            dict(row)
            for row in self.messages
            if str(row["conversationId"]) == str(conversation_id)
        ]
        rows.sort(key=lambda row: int(row.get("sequence") or 0))
        return rows

    def find_conversation_summary(self, conversation_id: str) -> dict | None:
        for row in self.summaries:
            if str(row["conversationId"]) == str(conversation_id):
                return dict(row)
        return None

    def upsert_conversation_summary(self, conversation_id: str, doc: dict) -> dict:
        with self._lock:
            payload = dict(doc)
            payload["conversationId"] = conversation_id
            for index, row in enumerate(self.summaries):
                if str(row["conversationId"]) == str(conversation_id):
                    merged = dict(row)
                    merged.update(payload)
                    self.summaries[index] = merged
                    return dict(merged)
            payload["_id"] = uuid4().hex
            self.summaries.append(payload)
            return dict(payload)


def _seed(
    repo: FakeRepo,
    *,
    status: str = "COMPLETED",
    user_id: str = "u1",
    messages: list[tuple[str, str]] | None = None,
) -> str:
    cid = uuid4().hex
    repo.sessions.append(
        {
            "_id": cid,
            "userId": user_id,
            "topicId": "t1",
            "status": status,
        }
    )
    if messages is None:
        messages = [
            ("user", "Hi, my name is Aayush."),
            ("assistant", "Nice to meet you, Aayush! Tell me about yourself."),
            ("user", "I am a software developer. I am working here since two years."),
            ("assistant", "That's interesting. What kind of software do you build?"),
            ("user", "I mostly build web applications."),
        ]
    for index, (role, content) in enumerate(messages, start=1):
        repo.messages.append(
            {
                "conversationId": cid,
                "role": role,
                "content": content,
                "sequence": index,
            }
        )
    return cid


def test_completed_conversation_creates_summary():
    repo = FakeRepo()
    analyzer = FakeAnalyzer()
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo)

    saved = svc.generateConversationSummary(cid, userId="u1")

    assert saved["summaryStatus"] == "COMPLETED"
    assert saved["conversationId"] == cid
    assert saved["summary"]
    assert saved["conversationMetrics"]["userMessageCount"] == 3
    assert saved["conversationMetrics"]["assistantMessageCount"] == 2
    assert saved["conversationMetrics"]["estimatedUserWords"] > 0
    assert len(repo.summaries) == 1
    assert repo.messages


def test_failed_conversation_does_not_generate_summary():
    repo = FakeRepo()
    analyzer = FakeAnalyzer()
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo, status="FAILED")

    with pytest.raises(ConversationNotCompleted) as exc:
        svc.generateConversationSummary(cid, userId="u1")
    assert exc.value.code == "CONVERSATION_NOT_COMPLETED"
    assert repo.summaries == []
    assert analyzer.calls == 0
    assert repo.sessions[0]["status"] == "FAILED"


def test_empty_conversation_is_insufficient():
    repo = FakeRepo()
    analyzer = FakeAnalyzer()
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo, messages=[])

    with pytest.raises(InsufficientConversationData) as exc:
        svc.generateConversationSummary(cid, userId="u1")
    assert exc.value.code == "INSUFFICIENT_CONVERSATION_DATA"
    assert analyzer.calls == 0
    assert repo.summaries == []


def test_goal_analysis_statuses():
    repo = FakeRepo()
    svc = ConversationSummaryService(repo, analyzer=FakeAnalyzer())
    cid = _seed(repo)

    saved = svc.generateConversationSummary(cid, userId="u1")
    by_id = {row["goalId"]: row["status"] for row in saved["goals"]}
    assert by_id["name"] == "COMPLETED"
    assert by_id["location"] == "PARTIAL"
    assert by_id["hobbies"] == "NOT_ATTEMPTED"


def test_mistake_detection_extracts_grammar():
    repo = FakeRepo()
    svc = ConversationSummaryService(repo, analyzer=FakeAnalyzer())
    cid = _seed(repo)

    saved = svc.generateConversationSummary(cid, userId="u1")
    mistakes = saved["mistakes"]
    assert mistakes
    assert mistakes[0]["type"] == "GRAMMAR"
    assert "since two years" in mistakes[0]["userText"]
    assert "for two years" in mistakes[0]["correction"]


def test_does_not_keep_hallucinated_facts():
    repo = FakeRepo()
    analyzer = FakeAnalyzer(
        _analysis(
            importantFacts=[
                {"fact": "User name is Aayush.", "confidence": 0.99},
                {"fact": "Aayush lives in Bhopal.", "confidence": 0.9},
            ]
        )
    )
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo, messages=[("user", "My name is Aayush.")])

    saved = svc.generateConversationSummary(cid, userId="u1")
    facts = " ".join(item["fact"] for item in saved["importantFacts"]).lower()
    assert "aayush" in facts
    assert "bhopal" not in facts


def test_duplicate_generation_returns_one_summary():
    repo = FakeRepo()
    analyzer = FakeAnalyzer()
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo)

    first = svc.generateConversationSummary(cid, userId="u1")
    second = svc.generateConversationSummary(cid, userId="u1")

    assert first["summaryId"] == second["summaryId"]
    assert len(repo.summaries) == 1
    assert analyzer.calls == 1


def test_invalid_json_does_not_store_corrupted_summary():
    repo = FakeRepo()
    analyzer = FakeAnalyzer()

    def _bad(*, system: str, user: str):
        analyzer.calls += 1
        return "not-json {"

    analyzer.analyze_json = _bad  # type: ignore[method-assign]
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo)

    with pytest.raises(SummaryGenerationFailed) as exc:
        svc.generateConversationSummary(cid, userId="u1")
    assert exc.value.code == "SUMMARY_GENERATION_FAILED"
    assert analyzer.calls == 3
    stored = repo.find_conversation_summary(cid)
    assert stored is None or stored.get("summaryStatus") == "FAILED"
    assert stored is None or not stored.get("goals")
    assert repo.sessions[0]["status"] == "COMPLETED"


def test_llm_failure_keeps_session_completed():
    repo = FakeRepo()
    analyzer = FakeAnalyzer(error=RuntimeError("llm timeout"))
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo)

    with pytest.raises(SummaryGenerationFailed) as exc:
        svc.generateConversationSummary(cid, userId="u1")
    assert exc.value.code == "SUMMARY_GENERATION_FAILED"
    assert analyzer.calls == 3
    assert repo.sessions[0]["status"] == "COMPLETED"
    stored = repo.find_conversation_summary(cid)
    assert stored is not None
    assert stored["summaryStatus"] == "FAILED"
    assert stored.get("summary") is None


def test_user_cannot_generate_or_read_another_users_summary():
    repo = FakeRepo()
    svc = ConversationSummaryService(repo, analyzer=FakeAnalyzer())
    cid = _seed(repo, user_id="u1")

    with pytest.raises(ConversationAccessDenied) as gen_exc:
        svc.generateConversationSummary(cid, userId="u2")
    assert gen_exc.value.code == "CONVERSATION_ACCESS_DENIED"

    svc.generateConversationSummary(cid, userId="u1")
    with pytest.raises(ConversationAccessDenied) as read_exc:
        svc.getConversationSummary(cid, "u2")
    assert read_exc.value.code == "CONVERSATION_ACCESS_DENIED"
    owned = svc.getConversationSummary(cid, "u1")
    assert owned["conversationId"] == cid


def test_invalid_goal_status_does_not_fail_summary():
    repo = FakeRepo()
    analyzer = FakeAnalyzer(
        _analysis(
            goals=[
                {
                    "goalId": "name",
                    "status": "DONE",
                    "evidence": "User introduced himself as Aayush.",
                },
                {
                    "goalId": "location",
                    "status": "maybe later",
                    "evidence": "",
                },
            ]
        )
    )
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo)

    saved = svc.generateConversationSummary(cid, userId="u1")

    assert saved["summaryStatus"] == "COMPLETED"
    by_id = {row["goalId"]: row["status"] for row in saved["goals"]}
    assert by_id["name"] == "COMPLETED"
    assert by_id["location"] == "NOT_ATTEMPTED"
    assert by_id["hobbies"] == "NOT_ATTEMPTED"


def test_empty_llm_summary_uses_transcript_fallback():
    repo = FakeRepo()
    analyzer = FakeAnalyzer(_analysis(summary="   "))
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo, messages=[("user", "Hi, my name is Aayush.")])

    saved = svc.generateConversationSummary(cid, userId="u1")

    assert saved["summaryStatus"] == "COMPLETED"
    assert "Aayush" in saved["summary"]


def test_analyzer_recovers_after_transient_failure():
    repo = FakeRepo()
    analyzer = FakeAnalyzer()
    calls = {"n": 0}

    def _flaky(*, system: str, user: str):
        calls["n"] += 1
        analyzer.calls += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary llm timeout")
        return _analysis()

    analyzer.analyze_json = _flaky  # type: ignore[method-assign]
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo)

    saved = svc.generateConversationSummary(cid, userId="u1")

    assert saved["summaryStatus"] == "COMPLETED"
    assert analyzer.calls == 2


def test_hollow_analysis_is_not_stored_as_completed():
    repo = FakeRepo()
    analyzer = FakeAnalyzer(
        _analysis(
            summary="   ",
            keyPoints=[],
            goals=[],
            mistakes=[],
            corrections=[],
            strengths=[],
            weaknesses=[],
            importantFacts=[],
            vocabulary=[],
            grammarPatterns=[],
            fluencyObservations=[],
        )
    )
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo)

    with pytest.raises(SummaryGenerationFailed) as exc:
        svc.generateConversationSummary(cid, userId="u1")
    assert exc.value.code == "SUMMARY_GENERATION_FAILED"
    stored = repo.find_conversation_summary(cid)
    assert stored is None or stored.get("summaryStatus") == "FAILED"
    assert stored is None or not stored.get("summary")


def test_force_regenerates_completed_summary():
    repo = FakeRepo()
    analyzer = FakeAnalyzer(_analysis(summary="First pass with little detail."))
    svc = ConversationSummaryService(repo, analyzer=analyzer)
    cid = _seed(repo)
    first = svc.generateConversationSummary(cid, userId="u1")
    analyzer.payload = _analysis(summary="Learner described a concrete future goal.")
    skipped = svc.generateConversationSummary(cid, userId="u1")
    forced = svc.generateConversationSummary(cid, userId="u1", force=True)
    assert skipped["summary"] == first["summary"]
    assert forced["summary"] == "Learner described a concrete future goal."
    assert analyzer.calls == 2
