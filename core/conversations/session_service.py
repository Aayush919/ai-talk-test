"""One Mongo conversation_session per successfully connected AI call."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from core import call_log
from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationError,
    ConversationNotFound,
)
from core.topics.errors import (
    TopicNotFound,
    TopicProgressInternalError,
    UserNotFound,
)

ACTIVE = "ACTIVE"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CALL_TYPE_AI_COACH = "AI_COACH"
_TERMINAL = frozenset({COMPLETED, FAILED})

REASON_NORMAL_DISCONNECT = "NORMAL_DISCONNECT"
REASON_USER_ENDED_CALL = "USER_ENDED_CALL"
REASON_REMOTE_ENDED_CALL = "REMOTE_ENDED_CALL"
REASON_NETWORK_FAILURE = "NETWORK_FAILURE"
REASON_WEBRTC_FAILURE = "WEBRTC_FAILURE"
REASON_SERVER_FAILURE = "SERVER_FAILURE"
REASON_TIMEOUT = "TIMEOUT"
REASON_UNKNOWN_FAILURE = "UNKNOWN_FAILURE"

NORMAL_END_REASONS = frozenset(
    {
        REASON_NORMAL_DISCONNECT,
        REASON_USER_ENDED_CALL,
        REASON_REMOTE_ENDED_CALL,
        REASON_TIMEOUT,
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _duration_seconds(started_at: Any, ended_at: Any) -> int:
    start = _as_aware(started_at)
    end = _as_aware(ended_at)
    if start is None or end is None:
        return 0
    try:
        return max(0, int((end - start).total_seconds()))
    except Exception:
        return 0


def _options(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def topic_id_of(current: dict[str, Any] | None) -> Any:
    payload = current or {}
    topic = payload.get("topic") if isinstance(payload.get("topic"), dict) else {}
    return (
        topic.get("_id")
        or topic.get("topicId")
        or (payload.get("topicProgress") or {}).get("topicId")
    )


def public_session(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    cid = doc.get("conversationId") or doc.get("_id")
    return {
        "conversationId": str(cid),
        "userId": doc.get("userId"),
        "topicId": str(doc["topicId"]) if doc.get("topicId") is not None else None,
        "status": doc.get("status"),
        "callType": doc.get("callType") or CALL_TYPE_AI_COACH,
        "startedAt": doc.get("startedAt"),
        "endedAt": doc.get("endedAt"),
        "durationSeconds": max(0, int(doc.get("durationSeconds") or 0)),
        "reason": doc.get("endReason") or doc.get("reason"),
    }


class ConversationSessionRepo(Protocol):
    def find_user(self, user_id: str) -> dict[str, Any] | None: ...
    def find_topic(self, topic_id: Any) -> dict[str, Any] | None: ...
    def insert_conversation_session(self, doc: dict[str, Any]) -> dict[str, Any]: ...
    def find_conversation_session(self, conversation_id: str) -> dict[str, Any] | None: ...
    def close_conversation_session(
        self,
        conversation_id: str,
        *,
        status: str,
        ended_at: datetime,
        duration_seconds: int,
        end_reason: str | None = None,
    ) -> dict[str, Any] | None: ...


class ConversationSessionService:
    def __init__(self, repo: ConversationSessionRepo) -> None:
        self.repo = repo

    def createConversationSession(
        self,
        *,
        userId: str,
        topicId: Any,
        callType: str = CALL_TYPE_AI_COACH,
    ) -> dict[str, Any]:
        uid = (userId or "").strip()
        if not uid:
            raise UserNotFound()
        if topicId is None or str(topicId).strip() == "":
            raise TopicNotFound()
        try:
            if self.repo.find_user(uid) is None:
                raise UserNotFound()
            if self.repo.find_topic(topicId) is None:
                raise TopicNotFound()
            now = _utc_now()
            doc = {
                "userId": uid,
                "topicId": topicId,
                "status": ACTIVE,
                "callType": (callType or CALL_TYPE_AI_COACH).strip() or CALL_TYPE_AI_COACH,
                "startedAt": now,
                "endedAt": None,
                "durationSeconds": 0,
                "messageCount": 0,
                "createdAt": now,
                "updatedAt": now,
            }
            saved = self.repo.insert_conversation_session(doc)
            out = public_session(saved)
            if not out:
                raise TopicProgressInternalError()
            return out
        except (UserNotFound, TopicNotFound):
            raise
        except Exception as exc:  # noqa: BLE001
            raise TopicProgressInternalError(str(exc)) from exc

    def completeConversationSession(
        self,
        conversationId: str,
        options: dict[str, Any] | None = None,
        *,
        userId: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        opts = _options(options)
        return self._finish(
            conversationId,
            COMPLETED,
            user_id=(userId or opts.get("userId")),
            reason=reason or opts.get("reason") or REASON_NORMAL_DISCONNECT,
        )

    def failConversationSession(
        self,
        conversationId: str,
        reason: str | None = None,
        *,
        userId: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        opts = _options(options)
        return self._finish(
            conversationId,
            FAILED,
            user_id=(userId or opts.get("userId")),
            reason=reason or opts.get("reason") or REASON_UNKNOWN_FAILURE,
        )

    def _finish(
        self,
        conversation_id: str,
        status: str,
        *,
        user_id: Any = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        cid = str(conversation_id or "").strip()
        if not cid:
            raise ConversationNotFound()
        owner = str(user_id).strip() if user_id else ""
        why = str(reason or "").strip() or (
            REASON_NORMAL_DISCONNECT if status == COMPLETED else REASON_UNKNOWN_FAILURE
        )
        try:
            existing = self.repo.find_conversation_session(cid)
            if existing is None:
                raise ConversationNotFound()
            if owner and str(existing.get("userId") or "") != owner:
                raise ConversationAccessDenied()
            previous = existing.get("status")
            if previous in _TERMINAL:
                out = public_session(existing)
                if not out:
                    raise ConversationNotFound()
                self._log_close(
                    existing,
                    previous_status=previous,
                    new_status=previous,
                    duration=int(out.get("durationSeconds") or 0),
                    reason=why,
                    idempotent=True,
                )
                return out
            started = existing.get("startedAt")
            if _as_aware(started) is None:
                call_log.error(
                    "SESSION",
                    "startedAt missing; durationSeconds=0",
                    extra={"conversationId": cid, "userId": existing.get("userId")},
                )
            ended = _utc_now()
            duration = _duration_seconds(started, ended)
            updated = self.repo.close_conversation_session(
                cid,
                status=status,
                ended_at=ended,
                duration_seconds=duration,
                end_reason=why,
            )
            final = updated or self.repo.find_conversation_session(cid) or existing
            out = public_session(final)
            if not out:
                raise ConversationNotFound()
            self._log_close(
                final,
                previous_status=previous,
                new_status=out.get("status"),
                duration=out.get("durationSeconds") or duration,
                reason=why,
                idempotent=updated is None,
            )
            return out
        except ConversationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TopicProgressInternalError(str(exc)) from exc

    def _log_close(
        self,
        doc: dict[str, Any],
        *,
        previous_status: Any,
        new_status: Any,
        duration: Any,
        reason: str,
        idempotent: bool,
    ) -> None:
        cid = doc.get("conversationId") or doc.get("_id")
        extra = {
            "conversationId": str(cid) if cid is not None else None,
            "userId": doc.get("userId"),
            "topicId": str(doc["topicId"]) if doc.get("topicId") is not None else None,
            "previousStatus": previous_status,
            "newStatus": new_status,
            "durationSeconds": max(0, int(duration or 0)),
            "reason": reason,
        }
        if idempotent:
            call_log.info("SESSION", "conversation session already closed", extra=extra)
        else:
            call_log.info("SESSION", "conversation session closed", extra=extra)
