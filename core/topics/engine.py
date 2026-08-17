"""Topic Engine — what to practice next. Never generates the spoken reply."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core import call_log
from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationNotFound,
)
from core.topics.config import DEFAULT_TOPIC_ENGINE_CONFIG, TopicEngineConfig
from core.topics.errors import TopicNotFound, UserNotFound
from core.topics.progress_service import (
    COMPLETED,
    IN_PROGRESS,
    NOT_STARTED,
    TopicProgressService,
    _goal_keys,
)

START = "START"
RESUME = "RESUME"
CONTINUE = "CONTINUE"
COMPLETE = "COMPLETE"
NEXT = "NEXT"
REVISIT = "REVISIT"
REVIEW = "REVIEW"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _topic_id(value: Any) -> str:
    return str(value) if value is not None else ""


def goal_records(topic: dict[str, Any] | None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for goal in (topic or {}).get("goals") or []:
        if isinstance(goal, dict) and goal.get("key"):
            out.append(
                {
                    "key": str(goal["key"]),
                    "description": str(goal.get("description") or goal["key"]),
                }
            )
        elif isinstance(goal, str) and goal:
            out.append({"key": goal, "description": goal.replace("_", " ")})
    return out


def select_current_goal(
    topic: dict[str, Any] | None,
    progress: dict[str, Any] | None,
) -> tuple[str | None, int, str]:
    """First remaining curriculum goal. No LLM. No spoken text."""
    records = goal_records(topic)
    keys = [item["key"] for item in records]
    completed = {
        str(item)
        for item in ((progress or {}).get("goalsCompleted") or [])
        if str(item) in set(keys)
    }
    remaining = [
        str(item)
        for item in ((progress or {}).get("goalsRemaining") or keys)
        if str(item) in set(keys) and str(item) not in completed
    ]
    if not remaining and keys and not completed:
        remaining = list(keys)
    current = remaining[0] if remaining else None
    index = keys.index(current) if current in keys else 0
    description = ""
    for item in records:
        if item["key"] == current:
            description = item["description"]
            break
    return current, index, description


def topic_meets_completion(
    topic: dict[str, Any] | None,
    progress: dict[str, Any] | None,
    *,
    duration_seconds: int | None = None,
    use_criteria: bool = True,
) -> bool:
    keys = _goal_keys(topic or {})
    if not keys:
        return False
    if str((progress or {}).get("status") or "") == COMPLETED:
        return True
    completed = [
        key
        for key in keys
        if str(key) in {str(item) for item in ((progress or {}).get("goalsCompleted") or [])}
    ]
    if len(completed) >= len(keys):
        return True
    if not use_criteria:
        return False
    criteria = (topic or {}).get("completionCriteria") or {}
    minimum = int(criteria.get("minimumGoals") or len(keys))
    if len(completed) < min(minimum, len(keys)):
        return False
    min_seconds = criteria.get("minimumConversationSeconds")
    if min_seconds and duration_seconds is not None:
        return int(duration_seconds) >= int(min_seconds)
    return True


def start_action(progress: dict[str, Any] | None) -> str:
    row = progress or {}
    status = str(row.get("status") or "")
    if status == COMPLETED:
        return REVIEW
    if row.get("needsRevisit") and status == IN_PROGRESS:
        return REVISIT
    has_history = bool(
        row.get("lastConversationId")
        or int(row.get("progress") or 0) > 0
        or (row.get("goalsCompleted") or [])
        or int(row.get("attemptCount") or 0) > 0
    )
    if has_history:
        return RESUME
    return START


def build_topic_plan(
    topic: dict[str, Any] | None,
    progress: dict[str, Any] | None,
    *,
    action: str,
    next_topic: dict[str, Any] | None = None,
    initialized: bool = False,
) -> dict[str, Any]:
    current_goal, index, description = select_current_goal(topic, progress)
    remaining = list((progress or {}).get("goalsRemaining") or [])
    completed = list((progress or {}).get("goalsCompleted") or [])
    status = str((progress or {}).get("status") or NOT_STARTED)
    topic_completed = status == COMPLETED or action in {COMPLETE, NEXT}
    should_continue = action in {START, RESUME, CONTINUE, REVISIT} and not topic_completed
    return {
        "topic": topic,
        "topicProgress": progress,
        "initialized": initialized,
        "action": action,
        "topicId": _topic_id((topic or {}).get("_id") or (progress or {}).get("topicId")),
        "topicTitle": (topic or {}).get("title"),
        "topicLevel": (topic or {}).get("level"),
        "topicStatus": status,
        "progress": int((progress or {}).get("progress") or 0),
        "currentGoalId": current_goal,
        "currentGoalIndex": index,
        "currentGoalDescription": description,
        "goalsCompleted": completed,
        "goalsRemaining": remaining,
        "shouldContinueTopic": should_continue,
        "topicCompleted": topic_completed,
        "needsRevisit": bool((progress or {}).get("needsRevisit")),
        "nextTopicId": _topic_id((next_topic or {}).get("_id")) or None,
        "nextTopicTitle": (next_topic or {}).get("title"),
        "nextTopicSlug": (next_topic or {}).get("slug"),
    }


class TopicEngine:
    """Curriculum brain. Mongo remains the source of truth for progress."""

    def __init__(
        self,
        repo: Any,
        *,
        progress: TopicProgressService | None = None,
        config: TopicEngineConfig | dict | None = None,
    ) -> None:
        self.repo = repo
        self.progress = progress or TopicProgressService(repo)
        if isinstance(config, TopicEngineConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = TopicEngineConfig.from_mapping(config)
        else:
            self.config = DEFAULT_TOPIC_ENGINE_CONFIG

    def getPracticePlan(self, userId: str) -> dict[str, Any]:
        packed = self.progress.getOrInitializeCurrentTopic(userId)
        topic = packed.get("topic") or {}
        row = packed.get("topicProgress") or {}
        action = start_action(row)
        nxt = self._peek_next_topic(str(row.get("userId") or userId), row.get("topicId"), topic)
        plan = build_topic_plan(
            topic,
            row,
            action=action,
            next_topic=nxt,
            initialized=bool(packed.get("initialized")),
        )
        call_log.info(
            "TOPIC",
            "topic_plan",
            extra={
                "userId": userId,
                "action": action,
                "topicId": plan.get("topicId"),
                "currentGoalId": plan.get("currentGoalId"),
            },
        )
        return plan

    def evaluateAfterConversation(
        self,
        conversationId: str,
        *,
        userId: str | None = None,
    ) -> dict[str, Any] | None:
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
        if not user_id:
            raise UserNotFound()
        topic_id = session.get("topicId")
        topic = self.repo.find_topic(topic_id)
        if topic is None:
            raise TopicNotFound()
        row = self.repo.find_progress(user_id, topic_id)
        if row is None:
            packed = self.progress.getOrInitializeCurrentTopic(user_id)
            row = packed.get("topicProgress") or self.repo.find_progress(user_id, topic_id)
        if row is None:
            return None
        if _trim(row.get("engineEvaluatedConversationId")) == cid:
            return build_topic_plan(topic, row, action=CONTINUE)
        duration = session.get("durationSeconds")
        try:
            duration_seconds = int(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration_seconds = None
        attempt = int(row.get("attemptCount") or 0) + 1
        fields: dict[str, Any] = {
            "attemptCount": attempt,
            "engineEvaluatedConversationId": cid,
            "updatedAt": _utc_now(),
        }
        complete = topic_meets_completion(
            topic,
            row,
            duration_seconds=duration_seconds,
            use_criteria=self.config.use_completion_criteria,
        )
        action = CONTINUE
        next_topic = None
        if complete:
            fields["status"] = COMPLETED
            fields["completedAt"] = row.get("completedAt") or _utc_now()
            if self.config.mark_revisit_on_active_weakness and self._has_active_weakness(user_id):
                fields["needsRevisit"] = True
            action = COMPLETE
            call_log.info(
                "TOPIC",
                "topic_completed",
                extra={"userId": user_id, "topicId": _topic_id(topic_id), "conversationId": cid},
            )
        else:
            fields["status"] = IN_PROGRESS
            call_log.info(
                "TOPIC",
                "topic_continued",
                extra={"userId": user_id, "topicId": _topic_id(topic_id), "conversationId": cid},
            )
        saved = self._write_fields(user_id, topic_id, fields) or {**row, **fields}
        if complete:
            advanced = self._advance_after_complete(user_id, saved)
            advanced_row = (advanced or {}).get("topicProgress") or {}
            advanced_id = _topic_id(advanced_row.get("topicId"))
            if (
                advanced is not None
                and str(advanced_row.get("status") or "") == IN_PROGRESS
                and advanced_id
                and advanced_id != _topic_id(topic_id)
            ):
                action = NEXT
                next_topic = advanced.get("topic")
                saved = advanced_row
                topic = next_topic or topic
                call_log.info(
                    "TOPIC",
                    "topic_advanced",
                    extra={"userId": user_id, "topicId": advanced_id},
                )
            elif saved.get("needsRevisit"):
                call_log.info(
                    "TOPIC",
                    "topic_revisit_marked",
                    extra={"userId": user_id, "topicId": _topic_id(topic_id)},
                )
        nxt = next_topic or self._peek_next_topic(user_id, saved.get("topicId"), topic)
        return build_topic_plan(topic, saved, action=action, next_topic=nxt)

    def _has_active_weakness(self, user_id: str) -> bool:
        finder = getattr(self.repo, "find_learning_memory", None)
        if not callable(finder):
            return False
        learning = finder(user_id) or {}
        for row in learning.get("recurringMistakes") or []:
            if isinstance(row, dict) and str(row.get("status") or "").upper() == "ACTIVE":
                return True
        return False

    def _peek_next_topic(
        self, user_id: str, current_topic_id: Any, current_topic: dict[str, Any]
    ) -> dict[str, Any] | None:
        level = _trim(current_topic.get("level"))
        lister = getattr(self.repo, "list_active_topics", None)
        if not callable(lister) or not level:
            return None
        current = _topic_id(current_topic_id)
        for topic in lister(level) or []:
            tid = topic.get("_id")
            if _topic_id(tid) == current:
                continue
            row = self.repo.find_progress(user_id, tid)
            if row is None or str(row.get("status") or "") == NOT_STARTED:
                return topic
        return None

    def _advance_after_complete(self, user_id: str, completed_row: dict[str, Any]) -> dict[str, Any] | None:
        _ = completed_row
        existing = self.repo.list_progress(user_id)
        waiting: list[tuple[int, dict[str, Any]]] = []
        for row in existing:
            if str(row.get("status") or "") != NOT_STARTED:
                continue
            topic = self.repo.find_topic(row.get("topicId"))
            order = int((topic or {}).get("order") or 10_000)
            waiting.append((order, row))
        waiting.sort(key=lambda item: item[0])
        if not waiting:
            return None
        chosen = waiting[0][1]
        updated = self.repo.mark_in_progress(user_id, chosen["topicId"])
        packed = self.progress._pack(updated or chosen, initialized=False)
        return packed

    def _write_fields(self, user_id: str, topic_id: Any, fields: dict[str, Any]) -> dict[str, Any] | None:
        writer = getattr(self.repo, "update_progress_fields", None)
        if callable(writer):
            return writer(user_id, topic_id, fields)
        row = self.repo.find_progress(user_id, topic_id)
        if not isinstance(row, dict):
            return None
        row.update(fields)
        return dict(row)
