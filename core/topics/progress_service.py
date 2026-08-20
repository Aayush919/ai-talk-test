"""Current topic for a voice call — lazy init on successful connect only."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationNotCompleted,
    ConversationNotFound,
    SummaryNotFound,
)
from core.topics.errors import (
    TopicHasNoGoals,
    TopicNotFound,
    TopicProgressInternalError,
    TopicProgressUpdateFailed,
    TopicsNotFoundForLevel,
    UserEnglishLevelRequired,
    UserNotFound,
)

IN_PROGRESS = "IN_PROGRESS"
NOT_STARTED = "NOT_STARTED"
COMPLETED = "COMPLETED"
_LEVELS = frozenset({"A1", "A2", "B1", "B2", "C1"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _goal_keys(topic: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for goal in topic.get("goals") or []:
        if isinstance(goal, dict) and goal.get("key"):
            keys.append(str(goal["key"]))
        elif isinstance(goal, str) and goal:
            keys.append(goal)
    return keys


def _level_of(user: dict[str, Any]) -> str:
    raw = user.get("englishLevel")
    if raw is None:
        raw = user.get("english_level")
    return str(raw or "").strip().upper()


class TopicProgressRepo(Protocol):
    def find_user(self, user_id: str) -> dict[str, Any] | None: ...
    def find_in_progress(self, user_id: str) -> dict[str, Any] | None: ...
    def list_progress(self, user_id: str) -> list[dict[str, Any]]: ...
    def list_active_topics(self, level: str) -> list[dict[str, Any]]: ...
    def find_topic(self, topic_id: Any) -> dict[str, Any] | None: ...
    def upsert_progress(self, docs: list[dict[str, Any]]) -> None: ...
    def mark_in_progress(self, user_id: str, topic_id: Any) -> dict[str, Any] | None: ...
    def find_conversation_session(self, conversation_id: str) -> dict[str, Any] | None: ...
    def find_conversation_summary(self, conversation_id: str) -> dict[str, Any] | None: ...
    def find_progress(self, user_id: str, topic_id: Any) -> dict[str, Any] | None: ...
    def apply_progress_from_conversation(
        self,
        user_id: str,
        topic_id: Any,
        conversation_id: str,
        fields: dict[str, Any],
    ) -> dict[str, Any] | None: ...


class TopicProgressService:
    def __init__(self, repo: TopicProgressRepo) -> None:
        self.repo = repo

    def getOrInitializeCurrentTopic(self, user_id: str) -> dict[str, Any]:
        uid = (user_id or "").strip()
        if not uid:
            raise UserNotFound()
        try:
            current = self.repo.find_in_progress(uid)
            if current:
                return self._pack(current, initialized=False)
            existing = self.repo.list_progress(uid)
            if existing:
                return self._activate_next(uid, existing)
            return self._initialize(uid)
        except (UserNotFound, UserEnglishLevelRequired, TopicsNotFoundForLevel):
            raise
        except Exception as exc:  # noqa: BLE001
            raise TopicProgressInternalError(str(exc)) from exc

    def _initialize(self, user_id: str) -> dict[str, Any]:
        user = self.repo.find_user(user_id)
        if user is None:
            raise UserNotFound()
        level = _level_of(user)
        if not level or level not in _LEVELS:
            raise UserEnglishLevelRequired()
        topics = self.repo.list_active_topics(level)
        if not topics:
            raise TopicsNotFoundForLevel()
        now = _utc_now()
        docs: list[dict[str, Any]] = []
        for i, topic in enumerate(topics):
            first = i == 0
            doc: dict[str, Any] = {
                "userId": user_id,
                "topicId": topic["_id"],
                "status": IN_PROGRESS if first else NOT_STARTED,
                "progress": 0,
                "goalsCompleted": [],
                "goalsRemaining": _goal_keys(topic),
                "attemptCount": 0,
                "updatedAt": now,
            }
            if first:
                doc["startedAt"] = now
            docs.append(doc)
        self.repo.upsert_progress(docs)
        current = self.repo.find_in_progress(user_id)
        if current is None:
            raise TopicProgressInternalError("topic_progress write did not persist")
        return self._pack(current, initialized=True)

    def _activate_next(
        self, user_id: str, existing: list[dict[str, Any]]
    ) -> dict[str, Any]:
        waiting: list[tuple[int, dict[str, Any]]] = []
        for row in existing:
            if row.get("status") != NOT_STARTED:
                continue
            topic = self.repo.find_topic(row.get("topicId"))
            order = int((topic or {}).get("order") or 10_000)
            waiting.append((order, row))
        waiting.sort(key=lambda item: item[0])
        if waiting:
            chosen = waiting[0][1]
            updated = self.repo.mark_in_progress(user_id, chosen["topicId"])
            return self._pack(updated or chosen, initialized=False)
        revisit = self._choose_revisit(existing)
        if revisit is not None:
            opener = getattr(self.repo, "reopen_for_revisit", None)
            updated = opener(user_id, revisit["topicId"]) if callable(opener) else None
            if updated is None:
                revisit["status"] = IN_PROGRESS
                revisit["updatedAt"] = _utc_now()
                updated = revisit
            return self._pack(updated, initialized=False)
        current = existing[0]
        return self._pack(current, initialized=False)

    def _choose_revisit(self, existing: list[dict[str, Any]]) -> dict[str, Any] | None:
        candidates: list[tuple[datetime, dict[str, Any]]] = []
        for row in existing:
            if str(row.get("status") or "") != COMPLETED:
                continue
            if not row.get("needsRevisit"):
                continue
            stamp = row.get("completedAt") or row.get("updatedAt") or _utc_now()
            candidates.append((stamp, row))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _pack(self, progress: dict[str, Any], *, initialized: bool) -> dict[str, Any]:
        topic = self.repo.find_topic(progress.get("topicId"))
        if topic is None:
            raise TopicsNotFoundForLevel()
        return {
            "topic": topic,
            "topicProgress": progress,
            "initialized": initialized,
        }

    def updateTopicProgressFromSummary(
        self,
        conversationId: str,
        *,
        userId: str | None = None,
    ) -> dict[str, Any]:
        cid = str(conversationId or "").strip()
        if not cid:
            raise ConversationNotFound()
        owner = str(userId or "").strip() or None
        try:
            session = self.repo.find_conversation_session(cid)
            if session is None:
                raise ConversationNotFound()
            if owner and str(session.get("userId") or "") != owner:
                raise ConversationAccessDenied()
            if session.get("status") != COMPLETED:
                raise ConversationNotCompleted()
            summary = self.repo.find_conversation_summary(cid)
            if summary is None or (
                summary.get("summaryStatus")
                and summary.get("summaryStatus") != COMPLETED
            ):
                raise SummaryNotFound()
            topic_id = session.get("topicId")
            topic = self.repo.find_topic(topic_id)
            if topic is None:
                raise TopicNotFound()
            goal_keys = _goal_keys(topic)
            if not goal_keys:
                raise TopicHasNoGoals()
            user_id = str(session.get("userId") or "").strip()
            if not user_id:
                raise UserNotFound()
            progress = self.repo.find_progress(user_id, topic_id)
            if progress is None:
                self.getOrInitializeCurrentTopic(user_id)
                progress = self.repo.find_progress(user_id, topic_id)
            if progress is None:
                raise TopicProgressUpdateFailed()
            if cid in {
                str(item)
                for item in (progress.get("processedConversationIds") or [])
            } or str(progress.get("lastProcessedConversationId") or "") == cid:
                return self._public_progress(progress, topic)
            newly_completed = _completed_goal_ids(summary.get("goals"), goal_keys)
            previous_completed = {
                str(item)
                for item in (progress.get("goalsCompleted") or [])
                if str(item) in goal_keys
            }
            completed = [key for key in goal_keys if key in previous_completed or key in newly_completed]
            remaining = [key for key in goal_keys if key not in set(completed)]
            percent = int(round((len(completed) / len(goal_keys)) * 100))
            previous_status = str(progress.get("status") or NOT_STARTED)
            now = _utc_now()
            if previous_status == COMPLETED:
                status = COMPLETED
                completed_at = progress.get("completedAt") or now
            elif percent >= 100:
                status = COMPLETED
                completed_at = progress.get("completedAt") or now
            else:
                status = IN_PROGRESS
                completed_at = None
            started_at = progress.get("startedAt")
            if started_at is None:
                started_at = now
            fields = {
                "status": status,
                "progress": percent,
                "goalsCompleted": completed,
                "goalsRemaining": remaining,
                "lastConversationId": cid,
                "lastProcessedConversationId": cid,
                "lastProcessedAt": now,
                "startedAt": started_at,
                "completedAt": completed_at,
                "updatedAt": now,
            }
            updated = self.repo.apply_progress_from_conversation(
                user_id, topic_id, cid, fields
            )
            final = updated or self.repo.find_progress(user_id, topic_id) or progress
            return self._public_progress(final, topic)
        except (
            ConversationNotFound,
            ConversationAccessDenied,
            ConversationNotCompleted,
            SummaryNotFound,
            TopicNotFound,
            TopicHasNoGoals,
            UserNotFound,
            UserEnglishLevelRequired,
            TopicsNotFoundForLevel,
        ):
            raise
        except TopicProgressUpdateFailed:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TopicProgressUpdateFailed() from exc

    def updateFromPracticedCall(
        self,
        conversationId: str,
        runtime_snapshot: dict[str, Any] | None,
        *,
        userId: str | None = None,
    ) -> list[dict[str, Any]]:
        cid = str(conversationId or "").strip()
        if not cid:
            raise ConversationNotFound()
        owner = str(userId or "").strip() or None
        session = self.repo.find_conversation_session(cid)
        if session is None:
            raise ConversationNotFound()
        if owner and str(session.get("userId") or "") != owner:
            raise ConversationAccessDenied()
        if session.get("status") != COMPLETED:
            raise ConversationNotCompleted()
        user_id = str(session.get("userId") or "").strip()
        if not user_id:
            raise UserNotFound()
        rows = self._snapshots_from_runtime(runtime_snapshot, session)
        if not rows:
            raise TopicProgressUpdateFailed()
        if self.repo.find_progress(user_id, rows[0].get("topicId")) is None:
            self.getOrInitializeCurrentTopic(user_id)
        results: list[dict[str, Any]] = []
        last_complete: dict[str, Any] | None = None
        all_complete = True
        for snap in rows:
            topic_id = snap.get("topicId")
            topic = self.repo.find_topic(topic_id)
            if topic is None:
                continue
            goal_keys = _goal_keys(topic)
            if not goal_keys:
                continue
            progress = self.repo.find_progress(user_id, topic_id)
            if progress is None:
                continue
            newly = {
                str(item)
                for item in (snap.get("goalsCompleted") or [])
                if str(item) in set(goal_keys)
            }
            previous_completed = {
                str(item)
                for item in (progress.get("goalsCompleted") or [])
                if str(item) in set(goal_keys)
            }
            completed = [key for key in goal_keys if key in previous_completed or key in newly]
            remaining = [key for key in goal_keys if key not in set(completed)]
            percent = int(round((len(completed) / len(goal_keys)) * 100))
            previous_status = str(progress.get("status") or NOT_STARTED)
            now = _utc_now()
            if previous_status == COMPLETED or percent >= 100:
                status = COMPLETED
                completed_at = progress.get("completedAt") or now
            else:
                status = IN_PROGRESS
                completed_at = None
                all_complete = False
            started_at = progress.get("startedAt") or now
            fields = {
                "status": status,
                "progress": percent,
                "goalsCompleted": completed,
                "goalsRemaining": remaining,
                "lastConversationId": cid,
                "lastProcessedConversationId": cid,
                "lastProcessedAt": now,
                "startedAt": started_at,
                "completedAt": completed_at,
                "updatedAt": now,
            }
            updated = self.repo.apply_progress_from_conversation(
                user_id, topic_id, cid, fields
            )
            final = updated or self.repo.find_progress(user_id, topic_id) or progress
            results.append(self._public_progress(final, topic))
            if status == COMPLETED:
                last_complete = final
        if all_complete and last_complete is not None:
            try:
                from core.topics.engine import TopicEngine

                TopicEngine(self.repo, progress=self)._advance_after_complete(
                    user_id, last_complete
                )
            except Exception:  # noqa: BLE001
                pass
        return results

    def _snapshots_from_runtime(
        self,
        runtime_snapshot: dict[str, Any] | None,
        session: dict[str, Any],
    ) -> list[dict[str, Any]]:
        snapshot = runtime_snapshot or {}
        practiced = [
            dict(item)
            for item in (snapshot.get("practicedTopics") or [])
            if isinstance(item, dict) and item.get("topicId")
        ]
        current_id = str(snapshot.get("topicId") or session.get("topicId") or "")
        seen = {str(item.get("topicId")) for item in practiced}
        if current_id and current_id not in seen:
            practiced.append(
                {
                    "topicId": current_id,
                    "goalsCompleted": list(snapshot.get("goalsCompleted") or []),
                    "goalsRemaining": list(snapshot.get("goalsRemaining") or []),
                }
            )
        return practiced

    def getPracticePlan(self, user_id: str) -> dict[str, Any]:
        from core.topics.engine import TopicEngine

        return TopicEngine(self.repo, progress=self).getPracticePlan(user_id)

    def evaluateAfterConversation(
        self,
        conversationId: str,
        *,
        userId: str | None = None,
    ) -> dict[str, Any] | None:
        from core.topics.engine import TopicEngine

        return TopicEngine(self.repo, progress=self).evaluateAfterConversation(
            conversationId, userId=userId
        )

    def _public_progress(
        self, progress: dict[str, Any], topic: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "userId": progress.get("userId"),
            "topicId": str(progress["topicId"]) if progress.get("topicId") is not None else None,
            "topicTitle": topic.get("title"),
            "status": progress.get("status"),
            "progress": int(progress.get("progress") or 0),
            "goalsCompleted": list(progress.get("goalsCompleted") or []),
            "goalsRemaining": list(progress.get("goalsRemaining") or []),
            "attemptCount": int(progress.get("attemptCount") or 0),
            "lastConversationId": (
                str(progress["lastConversationId"])
                if progress.get("lastConversationId") is not None
                else None
            ),
            "startedAt": progress.get("startedAt"),
            "completedAt": progress.get("completedAt"),
            "updatedAt": progress.get("updatedAt"),
        }


def _completed_goal_ids(raw_goals: Any, topic_keys: list[str]) -> set[str]:
    allowed = set(topic_keys)
    completed: set[str] = set()
    if not isinstance(raw_goals, list):
        return completed
    for item in raw_goals:
        if not isinstance(item, dict):
            continue
        goal_id = str(item.get("goalId") or item.get("key") or "").strip()
        status = str(item.get("status") or "").strip().upper()
        if goal_id in allowed and status == COMPLETED:
            completed.add(goal_id)
    return completed


def getOrInitializeCurrentTopic(
    user_id: str, *, service: TopicProgressService
) -> dict[str, Any]:
    return service.getOrInitializeCurrentTopic(user_id)
