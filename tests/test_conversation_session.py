"""Conversation session lifecycle — one session per successful AI call."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from core.conversations import session_service as cs
from core.conversations.session_service import ConversationSessionService
from core.topics.errors import TopicNotFound, UserNotFound
from core.topics.progress_service import TopicProgressService


def _now():
    return datetime.now(timezone.utc)


def _topic(tid: str, slug: str, order: int, level: str = "A1") -> dict:
    return {
        "_id": tid,
        "title": slug.replace("-", " ").title(),
        "slug": slug,
        "level": level,
        "order": order,
        "isActive": True,
        "goals": [
            {"key": "name", "description": "User can introduce their name."},
            {"key": "location", "description": "User can talk about where they live."},
        ],
    }


class FakeRepo:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.topics: list[dict] = []
        self.progress: list[dict] = []
        self.sessions: list[dict] = []
        self._lock = threading.Lock()

    def find_user(self, user_id: str) -> dict | None:
        return self.users.get(user_id)

    def find_in_progress(self, user_id: str) -> dict | None:
        for row in self.progress:
            if row["userId"] == user_id and row["status"] == "IN_PROGRESS":
                return row
        return None

    def list_progress(self, user_id: str) -> list[dict]:
        return [row for row in self.progress if row["userId"] == user_id]

    def list_active_topics(self, level: str) -> list[dict]:
        rows = [
            t for t in self.topics if t.get("level") == level and t.get("isActive")
        ]
        rows.sort(key=lambda t: t["order"])
        return rows

    def find_topic(self, topic_id) -> dict | None:
        for topic in self.topics:
            if topic["_id"] == topic_id or str(topic["_id"]) == str(topic_id):
                return topic
        return None

    def upsert_progress(self, docs: list[dict]) -> None:
        with self._lock:
            for doc in docs:
                key = (doc["userId"], doc["topicId"])
                exists = any(
                    (row["userId"], row["topicId"]) == key for row in self.progress
                )
                if not exists:
                    self.progress.append(dict(doc))

    def mark_in_progress(self, user_id: str, topic_id) -> dict | None:
        with self._lock:
            now = _now()
            for row in self.progress:
                if (
                    row["userId"] == user_id
                    and row["topicId"] == topic_id
                    and row["status"] == "NOT_STARTED"
                ):
                    row["status"] = "IN_PROGRESS"
                    row["startedAt"] = now
                    row["updatedAt"] = now
                    return dict(row)
        return None

    def insert_conversation_session(self, doc: dict) -> dict:
        with self._lock:
            saved = dict(doc)
            saved["_id"] = uuid4().hex
            self.sessions.append(saved)
            return dict(saved)

    def find_conversation_session(self, conversation_id: str) -> dict | None:
        for row in self.sessions:
            if str(row["_id"]) == str(conversation_id):
                return dict(row)
        return None

    def close_conversation_session(
        self,
        conversation_id: str,
        *,
        status: str,
        ended_at,
        duration_seconds: int,
        end_reason: str | None = None,
    ) -> dict | None:
        with self._lock:
            for row in self.sessions:
                if str(row["_id"]) != str(conversation_id):
                    continue
                if row.get("status") != "ACTIVE":
                    return None
                row["status"] = status
                row["endedAt"] = ended_at
                row["durationSeconds"] = duration_seconds
                row["updatedAt"] = ended_at
                if end_reason:
                    row["endReason"] = end_reason
                return dict(row)
        return None


def _ready_repo() -> FakeRepo:
    repo = FakeRepo()
    repo.users["u1"] = {"_id": "u1", "englishLevel": "A1"}
    repo.topics = [
        _topic("t1", "self-introduction", 1),
        _topic("t2", "daily-routine", 2),
    ]
    return repo


def _start_after_connect(
    *,
    connected: bool,
    user_id: str,
    topic_svc: TopicProgressService,
    conv_svc: ConversationSessionService,
) -> dict | None:
    """Mirrors LiveCallBridge: create a session only after the socket is up."""
    if not connected:
        return None
    current = topic_svc.getOrInitializeCurrentTopic(user_id)
    return conv_svc.createConversationSession(
        userId=user_id,
        topicId=current["topic"]["_id"],
        callType="AI_COACH",
    )


def test_successful_call_creates_active_session():
    repo = _ready_repo()
    svc = ConversationSessionService(repo)

    created = svc.createConversationSession(
        userId="u1", topicId="t1", callType="AI_COACH"
    )

    assert created["status"] == "ACTIVE"
    assert created["conversationId"]
    assert created["userId"] == "u1"
    assert created["topicId"] == "t1"
    assert created["callType"] == "AI_COACH"
    assert created["startedAt"] is not None
    assert created["endedAt"] is None
    assert created["durationSeconds"] == 0
    assert len(repo.sessions) == 1


def test_session_topic_matches_current_topic():
    repo = _ready_repo()
    topic_svc = TopicProgressService(repo)
    conv_svc = ConversationSessionService(repo)

    current = topic_svc.getOrInitializeCurrentTopic("u1")
    created = conv_svc.createConversationSession(
        userId="u1",
        topicId=current["topic"]["_id"],
        callType="AI_COACH",
    )

    assert created["topicId"] == str(current["topic"]["_id"])
    assert created["topicId"] == str(current["topicProgress"]["topicId"])


def test_normal_hangup_completes_with_duration(monkeypatch):
    repo = _ready_repo()
    svc = ConversationSessionService(repo)
    started = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=327)
    clock = {"now": started}
    monkeypatch.setattr(cs, "_utc_now", lambda: clock["now"])

    created = svc.createConversationSession(
        userId="u1", topicId="t1", callType="AI_COACH"
    )
    clock["now"] = ended
    done = svc.completeConversationSession(created["conversationId"])

    assert done["status"] == "COMPLETED"
    assert done["conversationId"] == created["conversationId"]
    assert done["startedAt"] == started
    assert done["endedAt"] == ended
    assert done["durationSeconds"] == 327


def test_failed_connection_does_not_create_session():
    repo = _ready_repo()
    topic_svc = TopicProgressService(repo)
    conv_svc = ConversationSessionService(repo)

    result = _start_after_connect(
        connected=False,
        user_id="u1",
        topic_svc=topic_svc,
        conv_svc=conv_svc,
    )

    assert result is None
    assert repo.sessions == []
    assert repo.progress == []


def test_unexpected_disconnect_fails_active_session():
    repo = _ready_repo()
    svc = ConversationSessionService(repo)
    created = svc.createConversationSession(
        userId="u1", topicId="t1", callType="AI_COACH"
    )

    failed = svc.failConversationSession(created["conversationId"])

    assert failed["status"] == "FAILED"
    assert failed["endedAt"] is not None
    assert failed["durationSeconds"] >= 0
    stored = repo.find_conversation_session(created["conversationId"])
    assert stored is not None
    assert stored["status"] == "FAILED"


def test_double_completion_does_not_corrupt_session(monkeypatch):
    repo = _ready_repo()
    svc = ConversationSessionService(repo)
    started = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=40)
    later = ended + timedelta(seconds=90)
    clock = {"now": started}
    monkeypatch.setattr(cs, "_utc_now", lambda: clock["now"])

    created = svc.createConversationSession(
        userId="u1", topicId="t1", callType="AI_COACH"
    )
    clock["now"] = ended
    first = svc.completeConversationSession(created["conversationId"])
    clock["now"] = later
    second = svc.completeConversationSession(created["conversationId"])

    assert first["status"] == "COMPLETED"
    assert second["status"] == "COMPLETED"
    assert second["endedAt"] == first["endedAt"] == ended
    assert second["durationSeconds"] == first["durationSeconds"] == 40
    assert len(repo.sessions) == 1


def test_failed_session_cannot_be_completed():
    repo = _ready_repo()
    svc = ConversationSessionService(repo)
    created = svc.createConversationSession(
        userId="u1", topicId="t1", callType="AI_COACH"
    )
    failed = svc.failConversationSession(created["conversationId"])

    again = svc.completeConversationSession(created["conversationId"])

    assert failed["status"] == "FAILED"
    assert again["status"] == "FAILED"
    assert again["endedAt"] == failed["endedAt"]
    assert again["durationSeconds"] == failed["durationSeconds"]


def test_multiple_calls_get_distinct_conversation_ids():
    repo = _ready_repo()
    svc = ConversationSessionService(repo)

    first = svc.createConversationSession(
        userId="u1", topicId="t1", callType="AI_COACH"
    )
    second = svc.createConversationSession(
        userId="u1", topicId="t1", callType="AI_COACH"
    )

    assert first["conversationId"] != second["conversationId"]
    assert len(repo.sessions) == 2
    assert {row["status"] for row in repo.sessions} == {"ACTIVE"}


def test_missing_user_returns_user_not_found():
    repo = _ready_repo()
    svc = ConversationSessionService(repo)

    with pytest.raises(UserNotFound) as exc:
        svc.createConversationSession(userId="missing", topicId="t1")

    assert exc.value.code == "USER_NOT_FOUND"
    assert repo.sessions == []


def test_missing_topic_returns_topic_not_found():
    repo = _ready_repo()
    svc = ConversationSessionService(repo)

    with pytest.raises(TopicNotFound) as exc:
        svc.createConversationSession(userId="u1", topicId="missing-topic")

    assert exc.value.code == "TOPIC_NOT_FOUND"
    assert repo.sessions == []
