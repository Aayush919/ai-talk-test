"""Topic progress from conversation summaries — deterministic, idempotent."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.conversations.errors import (
    ConversationNotCompleted,
    SummaryNotFound,
)
from core.topics.errors import TopicHasNoGoals
from core.topics.progress_service import TopicProgressService


def _now():
    return datetime.now(timezone.utc)


def _topic(goals: list[str] | None = None) -> dict:
    keys = (
        [
            "introduce_self",
            "talk_about_work",
            "talk_about_hobbies",
            "talk_about_background",
            "talk_about_goals",
        ]
        if goals is None
        else goals
    )
    return {
        "_id": "t1",
        "title": "Introduction",
        "slug": "a1-introduction",
        "level": "A1",
        "order": 1,
        "isActive": True,
        "goals": [{"key": key, "description": key.replace("_", " ")} for key in keys],
    }


def _goals(*pairs: tuple[str, str]) -> list[dict]:
    return [{"goalId": key, "status": status, "evidence": ""} for key, status in pairs]


class FakeRepo:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {"u1": {"_id": "u1", "englishLevel": "A1"}}
        self.topics: list[dict] = [_topic()]
        self.progress: list[dict] = []
        self.sessions: list[dict] = []
        self.summaries: list[dict] = []
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
        return [t for t in self.topics if t.get("level") == level and t.get("isActive")]

    def find_topic(self, topic_id) -> dict | None:
        for topic in self.topics:
            if topic["_id"] == topic_id or str(topic["_id"]) == str(topic_id):
                return topic
        return None

    def upsert_progress(self, docs: list[dict]) -> None:
        with self._lock:
            for doc in docs:
                key = (doc["userId"], doc["topicId"])
                if not any((row["userId"], row["topicId"]) == key for row in self.progress):
                    self.progress.append(dict(doc))

    def mark_in_progress(self, user_id: str, topic_id) -> dict | None:
        with self._lock:
            for row in self.progress:
                if (
                    row["userId"] == user_id
                    and row["topicId"] == topic_id
                    and row["status"] == "NOT_STARTED"
                ):
                    row["status"] = "IN_PROGRESS"
                    row["startedAt"] = _now()
                    return dict(row)
        return None

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

    def find_progress(self, user_id: str, topic_id) -> dict | None:
        for row in self.progress:
            if row["userId"] == user_id and str(row["topicId"]) == str(topic_id):
                return dict(row)
        return None

    def apply_progress_from_conversation(
        self,
        user_id: str,
        topic_id,
        conversation_id: str,
        fields: dict,
    ) -> dict | None:
        with self._lock:
            for row in self.progress:
                if row["userId"] != user_id or str(row["topicId"]) != str(topic_id):
                    continue
                processed = [str(item) for item in (row.get("processedConversationIds") or [])]
                if str(conversation_id) in processed:
                    return None
                row.update(fields)
                row["attemptCount"] = int(row.get("attemptCount") or 0) + 1
                row["processedConversationIds"] = processed + [str(conversation_id)]
                return dict(row)
        return None


def _seed(
    repo: FakeRepo,
    *,
    goals: list[dict],
    status: str = "COMPLETED",
    progress: dict | None = None,
    summary_status: str = "COMPLETED",
) -> str:
    cid = uuid4().hex
    repo.sessions.append(
        {
            "_id": cid,
            "userId": "u1",
            "topicId": "t1",
            "status": status,
        }
    )
    if summary_status:
        repo.summaries.append(
            {
                "conversationId": cid,
                "userId": "u1",
                "topicId": "t1",
                "summaryStatus": summary_status,
                "goals": goals,
            }
        )
    keys = [g["key"] for g in repo.topics[0]["goals"]]
    base = {
        "userId": "u1",
        "topicId": "t1",
        "status": "IN_PROGRESS",
        "progress": 0,
        "goalsCompleted": [],
        "goalsRemaining": list(keys),
        "attemptCount": 0,
        "lastConversationId": None,
        "startedAt": _now(),
        "completedAt": None,
        "processedConversationIds": [],
    }
    if progress:
        base.update(progress)
    repo.progress.append(base)
    return cid


def test_first_completed_goal_is_twenty_percent():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(
        repo,
        goals=_goals(
            ("introduce_self", "COMPLETED"),
            ("talk_about_work", "NOT_ATTEMPTED"),
            ("talk_about_hobbies", "NOT_ATTEMPTED"),
            ("talk_about_background", "NOT_ATTEMPTED"),
            ("talk_about_goals", "NOT_ATTEMPTED"),
        ),
    )

    updated = svc.updateTopicProgressFromSummary(cid, userId="u1")

    assert updated["progress"] == 20
    assert updated["status"] == "IN_PROGRESS"
    assert updated["goalsCompleted"] == ["introduce_self"]
    assert updated["attemptCount"] == 1
    assert updated["lastConversationId"] == cid
    assert updated["completedAt"] is None


def test_multiple_completed_goals():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(
        repo,
        goals=_goals(
            ("introduce_self", "COMPLETED"),
            ("talk_about_work", "COMPLETED"),
            ("talk_about_hobbies", "COMPLETED"),
            ("talk_about_background", "NOT_ATTEMPTED"),
            ("talk_about_goals", "NOT_ATTEMPTED"),
        ),
    )

    updated = svc.updateTopicProgressFromSummary(cid)
    assert updated["progress"] == 60
    assert updated["status"] == "IN_PROGRESS"
    assert updated["goalsCompleted"] == [
        "introduce_self",
        "talk_about_work",
        "talk_about_hobbies",
    ]


def test_all_goals_completed_marks_topic_complete():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(
        repo,
        goals=_goals(
            ("introduce_self", "COMPLETED"),
            ("talk_about_work", "COMPLETED"),
            ("talk_about_hobbies", "COMPLETED"),
            ("talk_about_background", "COMPLETED"),
            ("talk_about_goals", "COMPLETED"),
        ),
    )

    updated = svc.updateTopicProgressFromSummary(cid)
    assert updated["progress"] == 100
    assert updated["status"] == "COMPLETED"
    assert updated["goalsRemaining"] == []
    assert updated["completedAt"] is not None


def test_partial_goal_stays_remaining():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(
        repo,
        goals=_goals(
            ("introduce_self", "PARTIAL"),
            ("talk_about_work", "NOT_ATTEMPTED"),
            ("talk_about_hobbies", "NOT_ATTEMPTED"),
            ("talk_about_background", "NOT_ATTEMPTED"),
            ("talk_about_goals", "NOT_ATTEMPTED"),
        ),
    )

    updated = svc.updateTopicProgressFromSummary(cid)
    assert "introduce_self" not in updated["goalsCompleted"]
    assert "introduce_self" in updated["goalsRemaining"]
    assert updated["progress"] == 0
    assert updated["status"] == "IN_PROGRESS"


def test_not_attempted_stays_incomplete():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(
        repo,
        goals=_goals(
            ("introduce_self", "NOT_ATTEMPTED"),
            ("talk_about_work", "NOT_ATTEMPTED"),
            ("talk_about_hobbies", "NOT_ATTEMPTED"),
            ("talk_about_background", "NOT_ATTEMPTED"),
            ("talk_about_goals", "NOT_ATTEMPTED"),
        ),
    )

    updated = svc.updateTopicProgressFromSummary(cid)
    assert updated["goalsCompleted"] == []
    assert updated["goalsRemaining"] == [
        "introduce_self",
        "talk_about_work",
        "talk_about_hobbies",
        "talk_about_background",
        "talk_about_goals",
    ]


def test_previously_completed_goal_does_not_regress():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(
        repo,
        goals=_goals(
            ("introduce_self", "NOT_ATTEMPTED"),
            ("talk_about_work", "COMPLETED"),
        ),
        progress={
            "goalsCompleted": ["introduce_self"],
            "goalsRemaining": [
                "talk_about_work",
                "talk_about_hobbies",
                "talk_about_background",
                "talk_about_goals",
            ],
            "progress": 20,
            "attemptCount": 1,
        },
    )

    updated = svc.updateTopicProgressFromSummary(cid)
    assert updated["goalsCompleted"] == ["introduce_self", "talk_about_work"]
    assert updated["progress"] == 40
    assert "introduce_self" not in updated["goalsRemaining"]


def test_partial_then_completed_adds_goal():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    first = _seed(repo, goals=_goals(("talk_about_hobbies", "PARTIAL")))
    svc.updateTopicProgressFromSummary(first)

    second = uuid4().hex
    repo.sessions.append(
        {"_id": second, "userId": "u1", "topicId": "t1", "status": "COMPLETED"}
    )
    repo.summaries.append(
        {
            "conversationId": second,
            "userId": "u1",
            "topicId": "t1",
            "summaryStatus": "COMPLETED",
            "goals": _goals(("talk_about_hobbies", "COMPLETED")),
        }
    )

    updated = svc.updateTopicProgressFromSummary(second)
    assert "talk_about_hobbies" in updated["goalsCompleted"]
    assert "talk_about_hobbies" not in updated["goalsRemaining"]
    assert updated["attemptCount"] == 2


def test_duplicate_processing_increments_attempt_once():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(repo, goals=_goals(("introduce_self", "COMPLETED")))

    first = svc.updateTopicProgressFromSummary(cid)
    second = svc.updateTopicProgressFromSummary(cid)

    assert first["attemptCount"] == 1
    assert second["attemptCount"] == 1
    assert second["goalsCompleted"] == first["goalsCompleted"]


def test_invalid_goal_id_is_ignored():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(
        repo,
        goals=_goals(
            ("introduce_self", "COMPLETED"),
            ("goal_xyz", "COMPLETED"),
        ),
    )

    updated = svc.updateTopicProgressFromSummary(cid)
    assert updated["goalsCompleted"] == ["introduce_self"]
    assert "goal_xyz" not in updated["goalsCompleted"]
    assert "goal_xyz" not in updated["goalsRemaining"]
    assert len(repo.topics[0]["goals"]) == 5


def test_failed_conversation_does_not_update_progress():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(repo, goals=_goals(("introduce_self", "COMPLETED")), status="FAILED")

    with pytest.raises(ConversationNotCompleted) as exc:
        svc.updateTopicProgressFromSummary(cid)
    assert exc.value.code == "CONVERSATION_NOT_COMPLETED"
    assert repo.progress[0]["attemptCount"] == 0
    assert repo.progress[0]["goalsCompleted"] == []


def test_missing_summary_returns_not_found():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = uuid4().hex
    repo.sessions.append(
        {"_id": cid, "userId": "u1", "topicId": "t1", "status": "COMPLETED"}
    )
    repo.progress.append(
        {
            "userId": "u1",
            "topicId": "t1",
            "status": "IN_PROGRESS",
            "progress": 0,
            "goalsCompleted": [],
            "goalsRemaining": [g["key"] for g in repo.topics[0]["goals"]],
            "attemptCount": 0,
        }
    )

    with pytest.raises(SummaryNotFound) as exc:
        svc.updateTopicProgressFromSummary(cid)
    assert exc.value.code == "SUMMARY_NOT_FOUND"


def test_topic_with_zero_goals():
    repo = FakeRepo()
    repo.topics = [_topic(goals=[])]
    svc = TopicProgressService(repo)
    cid = uuid4().hex
    repo.sessions.append(
        {"_id": cid, "userId": "u1", "topicId": "t1", "status": "COMPLETED"}
    )
    repo.summaries.append(
        {
            "conversationId": cid,
            "summaryStatus": "COMPLETED",
            "goals": _goals(("introduce_self", "COMPLETED")),
        }
    )
    repo.progress.append(
        {
            "userId": "u1",
            "topicId": "t1",
            "status": "IN_PROGRESS",
            "attemptCount": 0,
            "goalsCompleted": [],
        }
    )

    with pytest.raises(TopicHasNoGoals) as exc:
        svc.updateTopicProgressFromSummary(cid)
    assert exc.value.code == "TOPIC_HAS_NO_GOALS"
    assert repo.progress[0]["attemptCount"] == 0


def test_existing_user_progress_is_merged_not_reset():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cid = _seed(
        repo,
        goals=_goals(("talk_about_work", "COMPLETED")),
        progress={
            "goalsCompleted": ["introduce_self"],
            "goalsRemaining": [
                "talk_about_work",
                "talk_about_hobbies",
                "talk_about_background",
                "talk_about_goals",
            ],
            "progress": 20,
            "attemptCount": 1,
            "startedAt": started,
        },
    )

    updated = svc.updateTopicProgressFromSummary(cid)
    assert updated["goalsCompleted"] == ["introduce_self", "talk_about_work"]
    assert updated["progress"] == 40
    assert updated["attemptCount"] == 2
    assert updated["startedAt"] == started


def test_concurrent_processing_updates_once():
    repo = FakeRepo()
    svc = TopicProgressService(repo)
    cid = _seed(repo, goals=_goals(("introduce_self", "COMPLETED")))

    def run(_index: int) -> dict:
        return svc.updateTopicProgressFromSummary(cid)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(8)))

    counts = {row["attemptCount"] for row in results}
    assert counts == {1}
    assert repo.progress[0]["attemptCount"] == 1
    assert repo.progress[0]["goalsCompleted"] == ["introduce_self"]
    assert repo.progress[0]["goalsCompleted"].count("introduce_self") == 1


def test_practiced_call_updates_each_topic_and_keeps_current_in_progress():
    repo = FakeRepo()
    repo.topics = [
        _topic(),
        {
            "_id": "t2",
            "title": "Daily Routine",
            "slug": "a1-daily-routine",
            "level": "A1",
            "order": 2,
            "isActive": True,
            "goals": [
                {"key": "wake_up", "description": "wake"},
                {"key": "morning", "description": "morning"},
            ],
        },
        {
            "_id": "t3",
            "title": "Family",
            "slug": "a1-family",
            "level": "A1",
            "order": 3,
            "isActive": True,
            "goals": [{"key": "family_members", "description": "family"}],
        },
    ]
    svc = TopicProgressService(repo)
    svc.getOrInitializeCurrentTopic("u1")
    cid = uuid4().hex
    repo.sessions.append(
        {
            "_id": cid,
            "userId": "u1",
            "topicId": "t1",
            "status": "COMPLETED",
        }
    )
    snapshot = {
        "topicId": "t2",
        "goalsCompleted": ["wake_up"],
        "goalsRemaining": ["morning"],
        "practicedTopics": [
            {
                "topicId": "t1",
                "goalsCompleted": [
                    "introduce_self",
                    "talk_about_work",
                    "talk_about_hobbies",
                    "talk_about_background",
                    "talk_about_goals",
                ],
                "goalsRemaining": [],
                "status": "COMPLETED",
            }
        ],
    }
    rows = svc.updateFromPracticedCall(cid, snapshot, userId="u1")
    assert len(rows) == 2
    intro = repo.find_progress("u1", "t1")
    daily = repo.find_progress("u1", "t2")
    family = repo.find_progress("u1", "t3")
    assert intro["status"] == "COMPLETED"
    assert intro["progress"] == 100
    assert daily["status"] == "IN_PROGRESS"
    assert daily["goalsCompleted"] == ["wake_up"]
    assert daily["progress"] == 50
    assert family["status"] == "NOT_STARTED"
    assert repo.find_in_progress("u1")["topicId"] == "t2"
