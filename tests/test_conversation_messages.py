"""Conversation messages — final STT/LLM turns only, sequenced per call."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationNotActive,
    ConversationNotFound,
    EmptyMessage,
)
from core.conversations.message_service import ConversationMessageService
from core.conversations.session_service import ConversationSessionService


def _topic(tid: str = "t1") -> dict:
    return {
        "_id": tid,
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
        self.sessions: list[dict] = []
        self.messages: list[dict] = []
        self._lock = threading.Lock()

    def find_user(self, user_id: str) -> dict | None:
        return self.users.get(user_id)

    def find_topic(self, topic_id) -> dict | None:
        for topic in self.topics:
            if topic["_id"] == topic_id or str(topic["_id"]) == str(topic_id):
                return topic
        return None

    def insert_conversation_session(self, doc: dict) -> dict:
        with self._lock:
            saved = dict(doc)
            saved["_id"] = uuid4().hex
            saved.setdefault("messageCount", 0)
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
                if end_reason:
                    row["endReason"] = end_reason
                return dict(row)
        return None

    def claim_next_message_sequence(
        self, conversation_id: str, *, user_id: str | None = None
    ) -> dict | None:
        with self._lock:
            for row in self.sessions:
                if str(row["_id"]) != str(conversation_id):
                    continue
                if row.get("status") != "ACTIVE":
                    return None
                if user_id and str(row.get("userId") or "") != str(user_id):
                    return None
                row["messageCount"] = int(row.get("messageCount") or 0) + 1
                return dict(row)
        return None

    def insert_message(self, doc: dict) -> dict:
        with self._lock:
            key = (str(doc["conversationId"]), int(doc["sequence"]))
            for row in self.messages:
                if (str(row["conversationId"]), int(row["sequence"])) == key:
                    raise RuntimeError("duplicate sequence")
            saved = dict(doc)
            saved["_id"] = uuid4().hex
            self.messages.append(saved)
            return dict(saved)

    def list_messages(self, conversation_id: str) -> list[dict]:
        rows = [
            dict(row)
            for row in self.messages
            if str(row["conversationId"]) == str(conversation_id)
        ]
        rows.sort(key=lambda row: int(row.get("sequence") or 0))
        return rows


def _ready() -> tuple[FakeRepo, ConversationSessionService, ConversationMessageService]:
    repo = FakeRepo()
    repo.users["u1"] = {"_id": "u1", "englishLevel": "A1"}
    repo.users["u2"] = {"_id": "u2", "englishLevel": "A1"}
    repo.topics = [_topic()]
    return repo, ConversationSessionService(repo), ConversationMessageService(repo)


def _open_call(
    sessions: ConversationSessionService, *, user_id: str = "u1"
) -> dict:
    return sessions.createConversationSession(
        userId=user_id, topicId="t1", callType="AI_COACH"
    )


def test_user_message_is_first_in_conversation():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions)

    saved = messages.createConversationMessage(
        conversationId=call["conversationId"],
        role="user",
        content="My name is Aayush.",
        metadata={"source": "voice"},
        userId="u1",
    )

    assert saved["role"] == "user"
    assert saved["content"] == "My name is Aayush."
    assert saved["conversationId"] == call["conversationId"]
    assert saved["sequence"] == 1
    assert saved["userId"] == "u1"
    assert saved["topicId"] == "t1"
    assert _repo.messages[0]["role"] == "USER"


def test_assistant_message_is_second():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions)
    messages.createConversationMessage(
        conversationId=call["conversationId"],
        role="user",
        content="My name is Aayush.",
        userId="u1",
    )

    saved = messages.createConversationMessage(
        conversationId=call["conversationId"],
        role="assistant",
        content="Nice to meet you, Aayush! Tell me about yourself.",
        metadata={"source": "ai"},
        userId="u1",
    )

    assert saved["role"] == "assistant"
    assert saved["sequence"] == 2


def test_message_ordering_is_per_conversation():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions)
    cid = call["conversationId"]
    turns = [
        ("user", "Hi, my name is Aayush."),
        ("assistant", "Nice to meet you, Aayush! Tell me about yourself."),
        ("user", "I am a software developer."),
        ("assistant", "That's interesting. What kind of software do you build?"),
    ]
    for role, content in turns:
        messages.createConversationMessage(
            conversationId=cid, role=role, content=content, userId="u1"
        )

    rows = messages.getConversationMessages(cid, "u1")
    assert [row["sequence"] for row in rows] == [1, 2, 3, 4]
    assert [row["role"] for row in rows] == ["user", "assistant", "user", "assistant"]


def test_partial_stt_saves_only_final_transcript():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions)
    cid = call["conversationId"]

    assert (
        messages.recordUserTranscript(
            conversationId=cid, content="My", is_final=False, userId="u1"
        )
        is None
    )
    assert (
        messages.recordUserTranscript(
            conversationId=cid, content="My name", is_final=False, userId="u1"
        )
        is None
    )
    assert (
        messages.recordUserTranscript(
            conversationId=cid, content="My name is", is_final=False, userId="u1"
        )
        is None
    )
    saved = messages.recordUserTranscript(
        conversationId=cid,
        content="My name is Aayush.",
        is_final=True,
        userId="u1",
        metadata={"source": "voice", "sttProvider": "deepgram"},
    )

    rows = messages.getConversationMessages(cid, "u1")
    assert saved is not None
    assert len(rows) == 1
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "My name is Aayush."
    assert rows[0]["sequence"] == 1


def test_streaming_ai_saves_one_assistant_message():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions)
    cid = call["conversationId"]
    messages.recordUserTranscript(
        conversationId=cid,
        content="Hi",
        is_final=True,
        userId="u1",
    )

    saved = messages.recordAssistantReply(
        conversationId=cid,
        chunks=["Hello", "Hello, my", "Hello, my name is..."],
        userId="u1",
        metadata={"source": "ai"},
    )

    rows = messages.getConversationMessages(cid, "u1")
    assert saved["role"] == "assistant"
    assert saved["content"] == "Hello, my name is..."
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert len([row for row in rows if row["role"] == "assistant"]) == 1


def test_empty_messages_are_rejected():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions)
    cid = call["conversationId"]

    for content in (None, "", "   "):
        with pytest.raises(EmptyMessage) as exc:
            messages.createConversationMessage(
                conversationId=cid, role="user", content=content, userId="u1"
            )
        assert exc.value.code == "EMPTY_MESSAGE"
    assert messages.getConversationMessages(cid, "u1") == []


def test_user_cannot_add_or_read_another_users_conversation():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions, user_id="u1")
    cid = call["conversationId"]
    messages.createConversationMessage(
        conversationId=cid, role="user", content="Secret from A.", userId="u1"
    )

    with pytest.raises(ConversationAccessDenied) as add_exc:
        messages.createConversationMessage(
            conversationId=cid, role="user", content="Hijack", userId="u2"
        )
    assert add_exc.value.code == "CONVERSATION_ACCESS_DENIED"

    with pytest.raises(ConversationAccessDenied) as read_exc:
        messages.getConversationMessages(cid, "u2")
    assert read_exc.value.code == "CONVERSATION_ACCESS_DENIED"

    rows = messages.getConversationMessages(cid, "u1")
    assert len(rows) == 1
    assert rows[0]["content"] == "Secret from A."


def test_invalid_conversation_returns_not_found():
    _repo, _sessions, messages = _ready()

    with pytest.raises(ConversationNotFound) as exc:
        messages.createConversationMessage(
            conversationId="missing",
            role="user",
            content="Hello",
            userId="u1",
        )
    assert exc.value.code == "CONVERSATION_NOT_FOUND"


def test_completed_session_rejects_new_messages():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions)
    cid = call["conversationId"]
    messages.createConversationMessage(
        conversationId=cid, role="user", content="Hi", userId="u1"
    )
    sessions.completeConversationSession(cid)

    with pytest.raises(ConversationNotActive) as exc:
        messages.createConversationMessage(
            conversationId=cid,
            role="assistant",
            content="Too late",
            userId="u1",
        )
    assert exc.value.code == "CONVERSATION_NOT_ACTIVE"
    rows = messages.getConversationMessages(cid, "u1")
    assert len(rows) == 1
    assert rows[0]["content"] == "Hi"


def test_messages_are_isolated_across_calls():
    _repo, sessions, messages = _ready()
    first = _open_call(sessions)
    second = _open_call(sessions)
    c1, c2 = first["conversationId"], second["conversationId"]
    assert c1 != c2

    messages.createConversationMessage(
        conversationId=c1, role="user", content="Call one", userId="u1"
    )
    messages.createConversationMessage(
        conversationId=c2, role="user", content="Call two", userId="u1"
    )
    messages.createConversationMessage(
        conversationId=c2,
        role="assistant",
        content="Reply on call two",
        userId="u1",
    )

    one = messages.getConversationMessages(c1, "u1")
    two = messages.getConversationMessages(c2, "u1")
    assert [row["content"] for row in one] == ["Call one"]
    assert [row["content"] for row in two] == ["Call two", "Reply on call two"]
    assert [row["sequence"] for row in two] == [1, 2]


def test_concurrent_writes_do_not_duplicate_sequences():
    _repo, sessions, messages = _ready()
    call = _open_call(sessions)
    cid = call["conversationId"]

    def write(index: int) -> dict:
        role = "user" if index % 2 == 0 else "assistant"
        return messages.createConversationMessage(
            conversationId=cid,
            role=role,
            content=f"line {index}",
            userId="u1",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        saved = list(pool.map(write, range(20)))

    sequences = [row["sequence"] for row in saved]
    assert len(sequences) == 20
    assert sorted(sequences) == list(range(1, 21))
    assert len(set(sequences)) == 20
    rows = messages.getConversationMessages(cid, "u1")
    assert [row["sequence"] for row in rows] == list(range(1, 21))


def test_mongo_store_keeps_close_and_claim_as_separate_methods():
    import inspect

    from wrappers.mongo_store import MongoStore

    source = inspect.getsource(MongoStore)
    assert source.count("def close_conversation_session") == 1
    assert "def claim_next_message_sequence" in source
    assert MongoStore.close_conversation_session is not MongoStore.claim_next_message_sequence

