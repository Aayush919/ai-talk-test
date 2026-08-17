"""Persist final user/assistant turns for one active conversation_session."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationError,
    ConversationNotActive,
    ConversationNotFound,
    EmptyMessage,
    InvalidMessageRole,
)
from core.topics.errors import TopicProgressInternalError

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
_ROLES = frozenset({ROLE_USER, ROLE_ASSISTANT})
_STORE_ROLES = {"user": "USER", "assistant": "ASSISTANT", "system": "SYSTEM"}
ACTIVE = "ACTIVE"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_role(role: Any) -> str:
    return _trim(role).lower()


def fold_stream_chunks(chunks: list[Any] | None) -> str:
    """One final assistant string from streamed tokens (cumulative or deltas)."""
    parts = [_trim(chunk) for chunk in (chunks or [])]
    parts = [part for part in parts if part]
    if not parts:
        return ""
    cumulative = all(
        nxt.startswith(prev) for prev, nxt in zip(parts, parts[1:])
    )
    return parts[-1] if cumulative else "".join(parts)


def public_message(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    mid = doc.get("messageId") or doc.get("_id")
    return {
        "messageId": str(mid) if mid is not None else None,
        "conversationId": str(doc["conversationId"]) if doc.get("conversationId") is not None else None,
        "userId": doc.get("userId"),
        "topicId": str(doc["topicId"]) if doc.get("topicId") is not None else None,
        "role": _normalize_role(doc.get("role")),
        "content": doc.get("content"),
        "sequence": int(doc.get("sequence") or 0),
        "timestamp": doc.get("timestamp"),
        "metadata": doc.get("metadata") or {},
    }


class ConversationMessageRepo(Protocol):
    def find_conversation_session(self, conversation_id: str) -> dict[str, Any] | None: ...
    def claim_next_message_sequence(
        self, conversation_id: str, *, user_id: str | None = None
    ) -> dict[str, Any] | None: ...
    def insert_message(self, doc: dict[str, Any]) -> dict[str, Any]: ...
    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]: ...


class ConversationMessageService:
    def __init__(self, repo: ConversationMessageRepo) -> None:
        self.repo = repo

    def createConversationMessage(
        self,
        *,
        conversationId: str,
        role: str,
        content: Any,
        metadata: dict[str, Any] | None = None,
        userId: str | None = None,
    ) -> dict[str, Any]:
        cid = _trim(conversationId)
        if not cid:
            raise ConversationNotFound()
        text = _trim(content)
        if not text:
            raise EmptyMessage()
        normalized_role = _normalize_role(role)
        if normalized_role not in _ROLES:
            raise InvalidMessageRole()
        owner = _trim(userId) or None
        try:
            session = self.repo.find_conversation_session(cid)
            self._assert_writable(session, owner)
            claimed = self.repo.claim_next_message_sequence(cid, user_id=owner)
            if claimed is None:
                latest = self.repo.find_conversation_session(cid)
                self._assert_writable(latest, owner)
                raise ConversationNotActive()
            now = _utc_now()
            meta = dict(metadata) if isinstance(metadata, dict) else {}
            doc = {
                "conversationId": claimed.get("_id", cid),
                "userId": claimed.get("userId"),
                "topicId": claimed.get("topicId"),
                "role": _STORE_ROLES[normalized_role],
                "content": text,
                "sequence": int(claimed.get("messageCount") or 0),
                "timestamp": now,
                "metadata": meta,
                "createdAt": now,
                "updatedAt": now,
            }
            saved = self.repo.insert_message(doc)
            out = public_message(saved)
            if not out:
                raise TopicProgressInternalError("insert_message returned no public row")
            return out
        except ConversationError:
            raise
        except TopicProgressInternalError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TopicProgressInternalError(f"{type(exc).__name__}: {exc}") from exc

    def recordUserTranscript(
        self,
        *,
        conversationId: str,
        content: str,
        is_final: bool,
        userId: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """STT partials are ignored; only the accepted final transcript is stored."""
        if not is_final:
            return None
        return self.createConversationMessage(
            conversationId=conversationId,
            role=ROLE_USER,
            content=content,
            metadata=metadata or {"source": "voice"},
            userId=userId,
        )

    def recordAssistantReply(
        self,
        *,
        conversationId: str,
        content: str | None = None,
        chunks: list[Any] | None = None,
        userId: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store one assistant message from the final reply (or folded stream chunks)."""
        text = _trim(content) or fold_stream_chunks(chunks)
        return self.createConversationMessage(
            conversationId=conversationId,
            role=ROLE_ASSISTANT,
            content=text,
            metadata=metadata or {"source": "ai"},
            userId=userId,
        )

    def getConversationMessages(
        self, conversationId: str, userId: str
    ) -> list[dict[str, Any]]:
        cid = _trim(conversationId)
        uid = _trim(userId)
        if not cid:
            raise ConversationNotFound()
        if not uid:
            raise ConversationAccessDenied()
        try:
            session = self.repo.find_conversation_session(cid)
            if session is None:
                raise ConversationNotFound()
            if str(session.get("userId") or "") != uid:
                raise ConversationAccessDenied()
            rows = self.repo.list_messages(cid)
            out: list[dict[str, Any]] = []
            for row in rows:
                pub = public_message(row)
                if pub:
                    out.append(pub)
            return out
        except ConversationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TopicProgressInternalError(str(exc)) from exc

    def _assert_writable(self, session: dict[str, Any] | None, owner: str | None) -> None:
        if session is None:
            raise ConversationNotFound()
        if owner and str(session.get("userId") or "") != owner:
            raise ConversationAccessDenied()
        if session.get("status") != ACTIVE:
            raise ConversationNotActive()
