"""Topic Engine — curriculum decisions only. No spoken replies."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import uuid4

from core.topics.engine import (
    COMPLETE,
    CONTINUE,
    NEXT,
    RESUME,
    REVISIT,
    START,
    TopicEngine,
    select_current_goal,
    topic_meets_completion,
)
from core.topics.progress_service import TopicProgressService


def _now():
    return datetime.now(timezone.utc)


def _topic(tid: str, slug: str, order: int, *, minimum_goals: int = 4) -> dict:
    return {
        "_id": tid,
        "title": slug.replace("-", " ").title(),
        "slug": slug,
        "level": "A1",
        "order": order,
        "isActive": True,
        "userId": None,
        "completionCriteria": {
            "minimumGoals": minimum_goals,
            "minimumConversationSeconds": 300,
        },
        "goals": [
            {"key": "g1", "description": "User can introduce their name."},
            {"key": "g2", "description": "User can talk about where they live."},
            {"key": "g3", "description": "User can talk about study or work."},
            {"key": "g4", "description": "User can talk about hobbies."},
            {"key": "g5", "description": "User can talk about a future goal."},
        ],
    }


class FakeRepo:
    def __init__(self) -> None:
        self.users = {"u1": {"_id": "u1", "englishLevel": "A1"}}
        self.topics = [
            _topic("t1", "a1-introduction", 1),
            _topic("t2", "a1-daily-routine", 2),
            _topic("t3", "a1-hobbies", 3),
        ]
        self.progress: list[dict] = []
        self.sessions: list[dict] = []
        self.learning: list[dict] = []
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
        rows = [t for t in self.topics if t.get("level") == level and t.get("isActive")]
        rows.sort(key=lambda item: item["order"])
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
                    row["updatedAt"] = _now()
                    return row
        return None

    def find_progress(self, user_id: str, topic_id) -> dict | None:
        for row in self.progress:
            if row["userId"] == user_id and str(row["topicId"]) == str(topic_id):
                return row
        return None

    def find_conversation_session(self, conversation_id: str) -> dict | None:
        for row in self.sessions:
            if str(row["_id"]) == str(conversation_id):
                return dict(row)
        return None

    def find_learning_memory(self, user_id: str) -> dict | None:
        for row in self.learning:
            if str(row["userId"]) == str(user_id):
                return dict(row)
        return None

    def update_progress_fields(self, user_id: str, topic_id, fields: dict) -> dict | None:
        with self._lock:
            for row in self.progress:
                if row["userId"] == user_id and str(row["topicId"]) == str(topic_id):
                    row.update(fields)
                    return dict(row)
        return None

    def reopen_for_revisit(self, user_id: str, topic_id) -> dict | None:
        with self._lock:
            for row in self.progress:
                if (
                    row["userId"] == user_id
                    and str(row["topicId"]) == str(topic_id)
                    and row.get("status") == "COMPLETED"
                    and row.get("needsRevisit")
                ):
                    row["status"] = "IN_PROGRESS"
                    row["updatedAt"] = _now()
                    return dict(row)
        return None


def _engine() -> TopicEngine:
    repo = FakeRepo()
    return TopicEngine(repo, progress=TopicProgressService(repo))


def _session(repo: FakeRepo, *, topic_id: str = "t1", duration: int = 400) -> str:
    cid = str(uuid4())
    repo.sessions.append(
        {
            "_id": cid,
            "userId": "u1",
            "topicId": topic_id,
            "status": "COMPLETED",
            "durationSeconds": duration,
        }
    )
    return cid


def test_topics_are_global_not_copied_per_user():
    engine = _engine()
    plan = engine.getPracticePlan("u1")
    assert plan["action"] == START
    assert plan["topic"]["slug"] == "a1-introduction"
    assert "userId" not in plan["topic"] or plan["topic"].get("userId") in {None, ""}
    assert all("userId" not in topic or topic.get("userId") in {None, ""} for topic in engine.repo.topics)


def test_new_learner_starts_first_curriculum_topic_and_first_goal():
    engine = _engine()
    plan = engine.getPracticePlan("u1")
    assert plan["currentGoalId"] == "g1"
    assert plan["currentGoalDescription"]
    assert plan["shouldContinueTopic"] is True
    assert plan["topicCompleted"] is False
    in_progress = [row for row in engine.repo.progress if row["status"] == "IN_PROGRESS"]
    assert len(in_progress) == 1


def test_partial_topic_resumes_across_calls():
    engine = _engine()
    first = engine.getPracticePlan("u1")
    row = engine.repo.find_in_progress("u1")
    row["goalsCompleted"] = ["g1"]
    row["goalsRemaining"] = ["g2", "g3", "g4", "g5"]
    row["progress"] = 20
    row["lastConversationId"] = "c-old"
    second = engine.getPracticePlan("u1")
    assert second["action"] == RESUME
    assert second["topicId"] == first["topicId"]
    assert second["currentGoalId"] == "g2"
    assert second["initialized"] is False


def test_incomplete_call_continues_same_topic():
    engine = _engine()
    engine.getPracticePlan("u1")
    row = engine.repo.find_in_progress("u1")
    row["goalsCompleted"] = ["g1"]
    row["goalsRemaining"] = ["g2", "g3", "g4", "g5"]
    row["progress"] = 20
    cid = _session(engine.repo, duration=120)
    result = engine.evaluateAfterConversation(cid, userId="u1")
    assert result["action"] == CONTINUE
    assert result["topicId"] == "t1"
    assert engine.repo.find_in_progress("u1")["topicId"] == "t1"


def test_completed_topic_advances_to_next_global_topic():
    engine = _engine()
    engine.getPracticePlan("u1")
    row = engine.repo.find_in_progress("u1")
    row["goalsCompleted"] = ["g1", "g2", "g3", "g4", "g5"]
    row["goalsRemaining"] = []
    row["progress"] = 100
    cid = _session(engine.repo)
    result = engine.evaluateAfterConversation(cid, userId="u1")
    assert result["action"] == NEXT
    assert result["topic"]["slug"] == "a1-daily-routine"
    intro = engine.repo.find_progress("u1", "t1")
    assert intro["status"] == "COMPLETED"
    assert engine.repo.find_in_progress("u1")["topicId"] == "t2"


def test_completion_uses_topic_minimum_goals():
    engine = _engine()
    engine.getPracticePlan("u1")
    row = engine.repo.find_in_progress("u1")
    row["goalsCompleted"] = ["g1", "g2", "g3", "g4"]
    row["goalsRemaining"] = ["g5"]
    row["progress"] = 80
    topic = engine.repo.find_topic("t1")
    assert topic_meets_completion(topic, row, duration_seconds=400) is True
    cid = _session(engine.repo, duration=400)
    result = engine.evaluateAfterConversation(cid, userId="u1")
    assert result["action"] == NEXT
    assert engine.repo.find_progress("u1", "t1")["status"] == "COMPLETED"


def test_short_call_below_minimum_time_does_not_complete_on_partial_goals():
    engine = _engine()
    engine.getPracticePlan("u1")
    row = engine.repo.find_in_progress("u1")
    row["goalsCompleted"] = ["g1", "g2", "g3", "g4"]
    row["goalsRemaining"] = ["g5"]
    row["progress"] = 80
    cid = _session(engine.repo, duration=60)
    result = engine.evaluateAfterConversation(cid, userId="u1")
    assert result["action"] == CONTINUE
    assert engine.repo.find_progress("u1", "t1")["status"] == "IN_PROGRESS"


def test_active_weakness_marks_completed_topic_for_later_revisit():
    engine = _engine()
    engine.getPracticePlan("u1")
    row = engine.repo.find_in_progress("u1")
    row["goalsCompleted"] = ["g1", "g2", "g3", "g4", "g5"]
    row["goalsRemaining"] = []
    row["progress"] = 100
    engine.repo.learning.append(
        {
            "userId": "u1",
            "recurringMistakes": [
                {"skill": "past_tense", "status": "ACTIVE", "frequency": 3}
            ],
        }
    )
    cid = _session(engine.repo)
    engine.evaluateAfterConversation(cid, userId="u1")
    intro = engine.repo.find_progress("u1", "t1")
    assert intro["needsRevisit"] is True
    assert intro["status"] == "COMPLETED"
    assert engine.repo.find_in_progress("u1")["topicId"] == "t2"


def test_all_topics_done_reopens_revisit_on_next_call():
    engine = _engine()
    engine.getPracticePlan("u1")
    for row in engine.repo.progress:
        row["status"] = "COMPLETED"
        row["progress"] = 100
        row["goalsRemaining"] = []
    engine.repo.progress[0]["needsRevisit"] = True
    plan = engine.getPracticePlan("u1")
    assert plan["action"] == REVISIT
    assert plan["topicId"] == "t1"
    assert engine.repo.find_in_progress("u1")["topicId"] == "t1"


def test_select_current_goal_is_first_remaining():
    topic = _topic("t1", "a1-introduction", 1)
    current, index, description = select_current_goal(
        topic,
        {"goalsCompleted": ["g1", "g2"], "goalsRemaining": ["g3", "g4", "g5"]},
    )
    assert current == "g3"
    assert index == 2
    assert "study or work" in description


def test_engine_does_not_generate_spoken_text():
    engine = _engine()
    plan = engine.getPracticePlan("u1")
    assert "response" not in plan
    assert "coach_text" not in plan
    blob = str(plan)
    assert "What do you" not in blob
    assert plan["currentGoalDescription"].startswith("User can")


def test_evaluate_is_idempotent_for_the_same_conversation():
    engine = _engine()
    engine.getPracticePlan("u1")
    row = engine.repo.find_in_progress("u1")
    row["goalsCompleted"] = ["g1", "g2", "g3", "g4", "g5"]
    row["goalsRemaining"] = []
    row["progress"] = 100
    cid = _session(engine.repo)
    first = engine.evaluateAfterConversation(cid, userId="u1")
    second = engine.evaluateAfterConversation(cid, userId="u1")
    assert first["action"] == NEXT
    in_progress = [row for row in engine.repo.progress if row["status"] == "IN_PROGRESS"]
    assert len(in_progress) == 1
    assert second["action"] == CONTINUE
