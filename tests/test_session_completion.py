"""Call-end session completion — COMPLETED vs FAILED, idempotent, owned."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from core.conversations import session_service as cs
from core.conversations.errors import ConversationAccessDenied, ConversationNotFound
from core.conversations.session_service import (
    ConversationSessionService,
    REASON_NETWORK_FAILURE,
    REASON_USER_ENDED_CALL,
    _duration_seconds,
)
from core.topics.progress_service import TopicProgressService


def _topic() -> dict:
    return {
        "_id": "t1",
        "title": "Self Introduction",
        "slug": "self-introduction",
        "level": "A1",
        "order": 1,
        "isActive": True,
        "goals": [],
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

    def find_topic(self, topic_id) -> dict | None:
        for topic in self.topics:
            if topic["_id"] == topic_id or str(topic["_id"]) == str(topic_id):
                return topic
        return None

    def find_in_progress(self, user_id: str) -> dict | None:
        return None

    def list_progress(self, user_id: str) -> list[dict]:
        return []

    def list_active_topics(self, level: str) -> list[dict]:
        return [t for t in self.topics if t.get("level") == level]

    def upsert_progress(self, docs: list[dict]) -> None:
        return None

    def mark_in_progress(self, user_id: str, topic_id) -> dict | None:
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


def _ready() -> tuple[FakeRepo, ConversationSessionService]:
    repo = FakeRepo()
    repo.users["u1"] = {"_id": "u1", "englishLevel": "A1"}
    repo.users["u2"] = {"_id": "u2", "englishLevel": "A1"}
    repo.topics = [_topic()]
    return repo, ConversationSessionService(repo)


def _open(svc: ConversationSessionService, *, user_id: str = "u1") -> dict:
    return svc.createConversationSession(
        userId=user_id, topicId="t1", callType="AI_COACH"
    )


def test_normal_completion_sets_ended_at_and_duration(monkeypatch):
    _repo, svc = _ready()
    started = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=327)
    clock = {"now": started}
    monkeypatch.setattr(cs, "_utc_now", lambda: clock["now"])
    created = _open(svc)
    clock["now"] = ended

    done = svc.completeConversationSession(
        created["conversationId"],
        {"reason": REASON_USER_ENDED_CALL},
        userId="u1",
    )

    assert done["status"] == "COMPLETED"
    assert done["endedAt"] == ended
    assert done["durationSeconds"] == 327
    assert done["reason"] == REASON_USER_ENDED_CALL


def test_failed_session_sets_ended_at_and_duration(monkeypatch):
    _repo, svc = _ready()
    started = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=12)
    clock = {"now": started}
    monkeypatch.setattr(cs, "_utc_now", lambda: clock["now"])
    created = _open(svc)
    clock["now"] = ended

    failed = svc.failConversationSession(
        created["conversationId"],
        REASON_NETWORK_FAILURE,
        userId="u1",
    )

    assert failed["status"] == "FAILED"
    assert failed["endedAt"] == ended
    assert failed["durationSeconds"] == 12
    assert failed["reason"] == REASON_NETWORK_FAILURE


def test_double_completion_keeps_original_ended_at(monkeypatch):
    _repo, svc = _ready()
    started = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=40)
    later = ended + timedelta(seconds=90)
    clock = {"now": started}
    monkeypatch.setattr(cs, "_utc_now", lambda: clock["now"])
    created = _open(svc)
    clock["now"] = ended
    first = svc.completeConversationSession(created["conversationId"], userId="u1")
    clock["now"] = later
    second = svc.completeConversationSession(created["conversationId"], userId="u1")

    assert first["status"] == second["status"] == "COMPLETED"
    assert second["endedAt"] == first["endedAt"] == ended
    assert second["durationSeconds"] == first["durationSeconds"] == 40


def test_complete_does_not_revive_failed_session():
    _repo, svc = _ready()
    created = _open(svc)
    failed = svc.failConversationSession(
        created["conversationId"], REASON_NETWORK_FAILURE, userId="u1"
    )

    again = svc.completeConversationSession(
        created["conversationId"],
        {"reason": REASON_USER_ENDED_CALL},
        userId="u1",
    )

    assert failed["status"] == "FAILED"
    assert again["status"] == "FAILED"
    assert again["endedAt"] == failed["endedAt"]
    assert again["durationSeconds"] == failed["durationSeconds"]


def test_complete_nonexistent_session_returns_not_found():
    _repo, svc = _ready()
    with pytest.raises(ConversationNotFound) as exc:
        svc.completeConversationSession("missing", userId="u1")
    assert exc.value.code == "CONVERSATION_NOT_FOUND"


def test_user_cannot_complete_another_users_session():
    repo, svc = _ready()
    created = _open(svc, user_id="u1")

    with pytest.raises(ConversationAccessDenied) as exc:
        svc.completeConversationSession(created["conversationId"], userId="u2")
    assert exc.value.code == "CONVERSATION_ACCESS_DENIED"
    stored = repo.find_conversation_session(created["conversationId"])
    assert stored is not None
    assert stored["status"] == "ACTIVE"


def test_failed_connection_does_not_create_or_complete_session():
    repo, svc = _ready()
    topic_svc = TopicProgressService(repo)

    def start_if_connected(*, connected: bool):
        if not connected:
            return None
        current = topic_svc.getOrInitializeCurrentTopic("u1")
        return svc.createConversationSession(
            userId="u1", topicId=current["topic"]["_id"], callType="AI_COACH"
        )

    assert start_if_connected(connected=False) is None
    assert repo.sessions == []
    with pytest.raises(ConversationNotFound):
        svc.completeConversationSession("never-created", userId="u1")


def test_very_short_call_is_still_completed(monkeypatch):
    _repo, svc = _ready()
    started = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=2)
    clock = {"now": started}
    monkeypatch.setattr(cs, "_utc_now", lambda: clock["now"])
    created = _open(svc)
    clock["now"] = ended

    done = svc.completeConversationSession(created["conversationId"], userId="u1")

    assert done["status"] == "COMPLETED"
    assert done["durationSeconds"] == 2


def test_completing_one_call_does_not_affect_another():
    repo, svc = _ready()
    first = _open(svc)
    second = _open(svc)
    assert first["conversationId"] != second["conversationId"]

    done = svc.completeConversationSession(first["conversationId"], userId="u1")

    assert done["status"] == "COMPLETED"
    other = repo.find_conversation_session(second["conversationId"])
    assert other is not None
    assert other["status"] == "ACTIVE"
    assert other["endedAt"] is None


def test_duration_seconds_never_negative(monkeypatch):
    repo, svc = _ready()
    started = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=5)
    clock = {"now": started}
    monkeypatch.setattr(cs, "_utc_now", lambda: clock["now"])
    created = _open(svc)
    repo.sessions[0]["startedAt"] = ended + timedelta(seconds=30)
    clock["now"] = ended

    done = svc.completeConversationSession(created["conversationId"], userId="u1")

    assert done["durationSeconds"] == 0
    assert _duration_seconds(ended + timedelta(seconds=10), ended) == 0
    assert _duration_seconds(None, ended) == 0


def test_duplicate_disconnect_events_keep_one_final_state(monkeypatch):
    repo, svc = _ready()
    started = datetime(2026, 8, 17, 10, 0, 0, tzinfo=timezone.utc)
    ended = started + timedelta(seconds=18)
    later = ended + timedelta(seconds=4)
    clock = {"now": started}
    monkeypatch.setattr(cs, "_utc_now", lambda: clock["now"])
    created = _open(svc)
    cid = created["conversationId"]
    clock["now"] = ended

    first = svc.completeConversationSession(
        cid, {"reason": REASON_USER_ENDED_CALL}, userId="u1"
    )
    clock["now"] = later
    second = svc.completeConversationSession(
        cid, {"reason": REASON_USER_ENDED_CALL}, userId="u1"
    )
    third = svc.failConversationSession(
        cid, REASON_NETWORK_FAILURE, userId="u1"
    )

    assert first["status"] == second["status"] == third["status"] == "COMPLETED"
    assert second["endedAt"] == third["endedAt"] == first["endedAt"] == ended
    assert first["durationSeconds"] == second["durationSeconds"] == 18
    assert len(repo.sessions) == 1
    assert repo.sessions[0]["endReason"] == REASON_USER_ENDED_CALL
