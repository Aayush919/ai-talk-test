"""Topic progress — first successful call init. No live Mongo required."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from core.topics.errors import (
    TopicsNotFoundForLevel,
    UserEnglishLevelRequired,
    UserNotFound,
)
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
            {"key": "education_or_work", "description": "study or work"},
            {"key": "hobbies", "description": "hobbies"},
            {"key": "future_goal", "description": "future goal"},
        ],
    }


class FakeRepo:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {}
        self.topics: list[dict] = []
        self.progress: list[dict] = []
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
            if topic["_id"] == topic_id:
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
            for row in self.progress:
                if (
                    row["userId"] == user_id
                    and row["topicId"] == topic_id
                    and row["status"] == "NOT_STARTED"
                ):
                    row["status"] = "IN_PROGRESS"
                    row["startedAt"] = _now()
                    row["updatedAt"] = _now()
                    return row
        return None


def _a1_repo() -> FakeRepo:
    repo = FakeRepo()
    repo.topics = [
        _topic("t1", "a1-introduction", 1),
        _topic("t2", "a1-daily-routine", 2),
        _topic("t3", "a1-family-friends", 3),
        _topic("t4", "a1-hobbies-interests", 4),
        _topic("t5", "a1-food-drinks", 5),
    ]
    repo.users["u1"] = {"_id": "u1", "englishLevel": "A1"}
    return repo


def test_new_user_first_successful_call() -> None:
    svc = TopicProgressService(_a1_repo())
    result = svc.getOrInitializeCurrentTopic("u1")
    assert result["initialized"] is True
    assert result["topic"]["slug"] == "a1-introduction"
    assert result["topicProgress"]["status"] == "IN_PROGRESS"
    rows = svc.repo.list_progress("u1")
    assert len(rows) == 5
    statuses = {row["topicId"]: row["status"] for row in rows}
    assert statuses["t1"] == "IN_PROGRESS"
    assert statuses["t2"] == "NOT_STARTED"
    assert statuses["t5"] == "NOT_STARTED"
    assert "userId" not in result["topic"]


def test_existing_user_first_ai_call_initializes() -> None:
    repo = _a1_repo()
    repo.users["old"] = {"_id": "old", "englishLevel": "A1"}
    svc = TopicProgressService(repo)
    result = svc.getOrInitializeCurrentTopic("old")
    assert result["initialized"] is True
    assert len(repo.list_progress("old")) == 5


def test_returning_user_no_new_records() -> None:
    repo = _a1_repo()
    svc = TopicProgressService(repo)
    svc.getOrInitializeCurrentTopic("u1")
    count = len(repo.progress)
    again = svc.getOrInitializeCurrentTopic("u1")
    assert again["initialized"] is False
    assert again["topic"]["slug"] == "a1-introduction"
    assert len(repo.progress) == count


def test_completed_current_activates_next() -> None:
    repo = _a1_repo()
    svc = TopicProgressService(repo)
    svc.getOrInitializeCurrentTopic("u1")
    intro = next(r for r in repo.progress if r["topicId"] == "t1")
    intro["status"] = "COMPLETED"
    result = svc.getOrInitializeCurrentTopic("u1")
    assert result["topic"]["slug"] == "a1-daily-routine"
    assert result["topicProgress"]["status"] == "IN_PROGRESS"
    intro = next(r for r in repo.progress if r["topicId"] == "t1")
    assert intro["status"] == "COMPLETED"


def test_multiple_calls_no_duplicates() -> None:
    repo = _a1_repo()
    svc = TopicProgressService(repo)
    for _ in range(8):
        svc.getOrInitializeCurrentTopic("u1")
    keys = [(r["userId"], r["topicId"]) for r in repo.progress]
    assert len(keys) == len(set(keys)) == 5


def test_concurrent_first_calls_no_duplicates() -> None:
    repo = _a1_repo()
    svc = TopicProgressService(repo)
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            svc.getOrInitializeCurrentTopic("u1")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    keys = [(r["userId"], r["topicId"]) for r in repo.progress]
    assert len(keys) == len(set(keys)) == 5
    in_prog = [r for r in repo.progress if r["status"] == "IN_PROGRESS"]
    assert len(in_prog) == 1


def test_missing_english_level() -> None:
    repo = _a1_repo()
    repo.users["u1"] = {"_id": "u1"}
    svc = TopicProgressService(repo)
    with pytest.raises(UserEnglishLevelRequired) as exc:
        svc.getOrInitializeCurrentTopic("u1")
    assert exc.value.code == "USER_ENGLISH_LEVEL_REQUIRED"


def test_no_active_topics() -> None:
    repo = FakeRepo()
    repo.users["u1"] = {"_id": "u1", "englishLevel": "C1"}
    svc = TopicProgressService(repo)
    with pytest.raises(TopicsNotFoundForLevel) as exc:
        svc.getOrInitializeCurrentTopic("u1")
    assert exc.value.code == "TOPICS_NOT_FOUND_FOR_LEVEL"


def test_user_not_found() -> None:
    svc = TopicProgressService(FakeRepo())
    with pytest.raises(UserNotFound) as exc:
        svc.getOrInitializeCurrentTopic("missing")
    assert exc.value.code == "USER_NOT_FOUND"


def test_b1_user_only_gets_b1_topics() -> None:
    repo = _a1_repo()
    repo.topics.append(_topic("b1", "b1-work-career", 1, level="B1"))
    repo.topics.append(_topic("b2", "b1-education", 2, level="B1"))
    repo.users["u1"] = {"_id": "u1", "englishLevel": "B1"}
    svc = TopicProgressService(repo)
    result = svc.getOrInitializeCurrentTopic("u1")
    assert result["topic"]["level"] == "B1"
    assert len(repo.list_progress("u1")) == 2
    assert all(svc.repo.find_topic(r["topicId"])["level"] == "B1" for r in repo.progress)
