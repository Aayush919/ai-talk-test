"""User profile memory — one doc per user, conservative, idempotent."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.conversations.errors import ConversationAccessDenied
from core.memory.errors import ProfileAccessDenied
from core.memory.profile_service import UserProfileMemoryService
from core.topics.progress_service import TopicProgressService


def _now():
    return datetime.now(timezone.utc)


def _candidates(*rows: dict) -> dict:
    return {"facts": list(rows)}


def _fact(key: str, value: str, confidence: float = 0.98, action: str = "UPSERT") -> dict:
    return {
        "key": key,
        "value": value,
        "confidence": confidence,
        "action": action,
    }


class FakeAnalyzer:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload if payload is not None else _candidates()
        self.calls = 0

    def analyze_json(self, *, system: str, user: str) -> dict:
        self.calls += 1
        return self.payload


class FakeRepo:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {"u1": {"_id": "u1"}, "u2": {"_id": "u2"}}
        self.sessions: list[dict] = []
        self.summaries: list[dict] = []
        self.profiles: list[dict] = []
        self.topics: list[dict] = [
            {
                "_id": "t1",
                "title": "Introduction",
                "slug": "a1-introduction",
                "level": "A1",
                "order": 1,
                "isActive": True,
                "goals": [
                    {"key": "g1", "description": "one"},
                    {"key": "g2", "description": "two"},
                    {"key": "g3", "description": "three"},
                    {"key": "g4", "description": "four"},
                    {"key": "g5", "description": "five"},
                ],
            }
        ]
        self.progress: list[dict] = []
        self._lock = threading.Lock()

    def find_user(self, user_id: str) -> dict | None:
        return self.users.get(user_id)

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

    def find_user_profile(self, user_id: str) -> dict | None:
        for row in self.profiles:
            if str(row["userId"]) == str(user_id):
                return dict(row)
        return None

    def apply_profile_from_conversation(
        self,
        user_id: str,
        conversation_id: str,
        fields: dict,
    ) -> dict | None:
        with self._lock:
            cid = str(conversation_id)
            for row in self.profiles:
                if str(row["userId"]) != str(user_id):
                    continue
                processed = [
                    str(item) for item in (row.get("processedConversationIds") or [])
                ]
                if cid in processed:
                    return None
                row.update(fields)
                row["version"] = int(row.get("version") or 0) + 1
                row["processedConversationIds"] = processed + [cid]
                return dict(row)
            doc = {
                "userId": user_id,
                "createdAt": fields.get("updatedAt") or _now(),
                "version": 1,
                "processedConversationIds": [cid],
            }
            doc.update(fields)
            self.profiles.append(doc)
            return dict(doc)

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
                processed = [
                    str(item) for item in (row.get("processedConversationIds") or [])
                ]
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
    user_id: str = "u1",
    summary: str = "The learner introduced themselves.",
    important_facts: list | None = None,
    goals: list | None = None,
) -> str:
    cid = uuid4().hex
    repo.sessions.append(
        {
            "_id": cid,
            "userId": user_id,
            "topicId": "t1",
            "status": "COMPLETED",
        }
    )
    repo.summaries.append(
        {
            "conversationId": cid,
            "userId": user_id,
            "topicId": "t1",
            "summaryStatus": "COMPLETED",
            "summary": summary,
            "importantFacts": important_facts or [],
            "goals": goals
            or [
                {"goalId": "g1", "status": "COMPLETED", "evidence": "name"},
                {"goalId": "g2", "status": "COMPLETED", "evidence": "work"},
                {"goalId": "g3", "status": "NOT_ATTEMPTED", "evidence": ""},
                {"goalId": "g4", "status": "NOT_ATTEMPTED", "evidence": ""},
                {"goalId": "g5", "status": "NOT_ATTEMPTED", "evidence": ""},
            ],
        }
    )
    return cid


def _stored_text(doc: dict) -> str:
    profile = doc.get("profile") or {}
    parts = [str(profile.get(key) or "") for key in profile]
    for key in ("hobbies", "interests", "goals", "communicationPreferences"):
        parts.extend(str(item) for item in (profile.get(key) or []))
    for fact in doc.get("facts") or []:
        parts.append(str(fact.get("key") or ""))
        parts.append(str(fact.get("value") or ""))
    return " ".join(parts).lower()


def test_new_profile_stores_explicit_name_and_profession():
    repo = FakeRepo()
    cid = _seed(
        repo,
        summary="My name is Aayush and I am a software developer.",
        important_facts=[
            {"fact": "User name is Aayush.", "confidence": 0.99},
            {"fact": "User is a software developer.", "confidence": 0.98},
        ],
    )
    analyzer = FakeAnalyzer(
        _candidates(
            _fact("name", "Aayush", 0.99),
            _fact("profession", "software developer", 0.98),
        )
    )
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert result["profile"]["name"] == "Aayush"
    assert result["profile"]["profession"] == "software developer"
    assert len(repo.profiles) == 1
    assert repo.profiles[0]["userId"] == "u1"


def test_summary_facts_are_stored_when_profile_llm_is_empty():
    repo = FakeRepo()
    cid = _seed(
        repo,
        summary="The learner introduced themselves and their work.",
        important_facts=[
            {"fact": "Learner's name is Ayush Manvi", "confidence": 1.0},
            {"fact": "The learner is a software developer.", "confidence": 0.95},
        ],
        goals=[
            {
                "goalId": "name",
                "status": "COMPLETED",
                "evidence": "My name is Ayush Manvi.",
            },
            {
                "goalId": "location",
                "status": "COMPLETED",
                "evidence": "I am living in Chandigarh.",
            },
            {
                "goalId": "education_or_work",
                "status": "COMPLETED",
                "evidence": "I am a software developer.",
            },
        ],
    )
    result = UserProfileMemoryService(
        repo, analyzer=FakeAnalyzer(_candidates())
    ).extractAndUpdateUserProfileMemory(cid)
    assert result["profile"]["name"] == "Ayush Manvi"
    assert result["profile"]["profession"] == "software developer"
    assert result["profile"]["location"] == "Chandigarh"
    keys = {row["key"] for row in result["facts"]}
    assert {"name", "profession", "location"} <= keys


def test_existing_profession_replaced_by_newer_explicit_statement():
    repo = FakeRepo()
    now = _now()
    repo.profiles.append(
        {
            "userId": "u1",
            "profile": {"profession": "teacher"},
            "facts": [
                {
                    "key": "profession",
                    "value": "teacher",
                    "confidence": 0.97,
                    "sourceConversationId": "old",
                    "firstSeenAt": now,
                    "lastConfirmedAt": now,
                }
            ],
            "version": 1,
            "processedConversationIds": ["old"],
            "createdAt": now,
            "updatedAt": now,
        }
    )
    cid = _seed(repo, summary="I now work as a software developer.")
    analyzer = FakeAnalyzer(
        _candidates(_fact("profession", "software developer", 0.98))
    )
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert result["profile"]["profession"] == "software developer"
    assert result["profile"]["profession"] != "teacher"
    keys = {row["key"]: row["value"] for row in result["facts"]}
    assert keys["profession"] == "software developer"
    assert keys.get("previous_profession") == "teacher"


def test_rejected_ai_location_is_not_stored():
    repo = FakeRepo()
    cid = _seed(
        repo,
        summary="AI asked if the user lives in Delhi. The user said no.",
        important_facts=[],
    )
    analyzer = FakeAnalyzer(
        _candidates(_fact("city", "Delhi", 0.95), _fact("name", "Delhi", 0.4))
    )
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert "delhi" not in _stored_text(result)
    assert not (result.get("profile") or {}).get("name")


def test_explicit_native_language():
    repo = FakeRepo()
    cid = _seed(repo, summary="My native language is Hindi.")
    analyzer = FakeAnalyzer(_candidates(_fact("nativeLanguage", "Hindi", 0.99)))
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert result["profile"]["nativeLanguage"] == "Hindi"


def test_duplicate_hobby_is_deduplicated():
    repo = FakeRepo()
    cid = _seed(repo, summary="I love playing cricket. Cricket is my favourite sport.")
    analyzer = FakeAnalyzer(
        _candidates(
            _fact("hobby", "cricket", 0.95),
            _fact("hobby", "Cricket", 0.96),
        )
    )
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert result["profile"]["hobbies"] == ["cricket"]
    hobby_facts = [row for row in result["facts"] if row["key"] == "hobby"]
    assert len(hobby_facts) == 1


def test_temporary_mood_is_not_stored():
    repo = FakeRepo()
    cid = _seed(repo, summary="The user said they are tired today.")
    analyzer = FakeAnalyzer(
        _candidates(
            _fact("preferredLearningStyle", "I am tired today", 0.99),
            _fact("mood", "tired", 0.99),
        )
    )
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert "tired" not in _stored_text(result)
    assert result["profile"] == {}
    assert result["facts"] == []


def test_topic_progress_is_not_written_to_profile_memory():
    repo = FakeRepo()
    repo.users["u1"] = {"_id": "u1", "englishLevel": "A1"}
    cid = _seed(
        repo,
        summary="The learner completed 2 of 5 Introduction goals.",
        goals=[
            {"goalId": "g1", "status": "COMPLETED", "evidence": "ok"},
            {"goalId": "g2", "status": "COMPLETED", "evidence": "ok"},
            {"goalId": "g3", "status": "NOT_ATTEMPTED", "evidence": ""},
            {"goalId": "g4", "status": "NOT_ATTEMPTED", "evidence": ""},
            {"goalId": "g5", "status": "NOT_ATTEMPTED", "evidence": ""},
        ],
    )
    progress = TopicProgressService(repo).updateTopicProgressFromSummary(cid)
    analyzer = FakeAnalyzer(
        _candidates(_fact("goal", "2/5 goals completed", 0.99))
    )
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert progress["progress"] == 40
    assert "g1" in progress["goalsCompleted"]
    blob = _stored_text(result)
    assert "2/5" not in blob
    assert "goals completed" not in blob
    assert result["profile"].get("goals") in (None, [])


def test_learning_mistakes_are_not_profile_memory():
    repo = FakeRepo()
    cid = _seed(
        repo,
        summary="The user repeatedly made article mistakes.",
    )
    analyzer = FakeAnalyzer(
        _candidates(
            _fact("preferredLearningStyle", "User often makes article mistakes", 0.99),
            _fact("education", "grammar mistake with articles", 0.99),
        )
    )
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert "mistake" not in _stored_text(result)
    assert "grammar" not in _stored_text(result)
    assert result["facts"] == []


def test_duplicate_processing_does_not_duplicate_memory_or_change_timestamps():
    repo = FakeRepo()
    cid = _seed(repo, summary="My name is Aayush.")
    analyzer = FakeAnalyzer(_candidates(_fact("name", "Aayush", 0.99)))
    svc = UserProfileMemoryService(repo, analyzer=analyzer)
    first = svc.extractAndUpdateUserProfileMemory(cid)
    stored = repo.profiles[0]
    first_seen = stored["facts"][0]["firstSeenAt"]
    last_confirmed = stored["facts"][0]["lastConfirmedAt"]
    version = stored["version"]
    updated_at = stored["updatedAt"]
    second = svc.extractAndUpdateUserProfileMemory(cid)
    assert analyzer.calls == 1
    assert second["profile"]["name"] == "Aayush"
    assert len(second["facts"]) == 1
    assert len(repo.profiles) == 1
    assert repo.profiles[0]["facts"][0]["firstSeenAt"] == first_seen
    assert repo.profiles[0]["facts"][0]["lastConfirmedAt"] == last_confirmed
    assert repo.profiles[0]["version"] == version
    assert repo.profiles[0]["updatedAt"] == updated_at
    assert first["version"] == second["version"]


def test_user_cannot_read_another_users_profile():
    repo = FakeRepo()
    now = _now()
    repo.profiles.append(
        {
            "userId": "u2",
            "profile": {"name": "B", "profession": "teacher"},
            "facts": [],
            "version": 1,
            "processedConversationIds": [],
            "createdAt": now,
            "updatedAt": now,
        }
    )
    svc = UserProfileMemoryService(repo, analyzer=FakeAnalyzer())
    with pytest.raises(ProfileAccessDenied) as denied:
        svc.getUserProfileMemory("u2", requesterId="u1")
    assert denied.value.code == "ACCESS_DENIED"
    own = svc.getUserProfileMemory("u2", requesterId="u2")
    assert own["profile"]["name"] == "B"
    cid = _seed(repo, user_id="u2", summary="ok")
    with pytest.raises(ConversationAccessDenied):
        svc.extractAndUpdateUserProfileMemory(cid, userId="u1")


def test_invalid_llm_key_is_ignored():
    repo = FakeRepo()
    cid = _seed(repo, summary="The user mentioned a car.")
    analyzer = FakeAnalyzer(_candidates(_fact("favorite_car", "BMW", 0.99)))
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert "bmw" not in _stored_text(result)
    assert "favorite_car" not in _stored_text(result)
    assert result["profile"] == {}
    assert result["facts"] == []


def test_low_confidence_is_not_persisted():
    repo = FakeRepo()
    cid = _seed(repo, summary="Maybe the user works in tech.")
    analyzer = FakeAnalyzer(
        _candidates(_fact("profession", "software developer", 0.45))
    )
    result = UserProfileMemoryService(repo, analyzer=analyzer).extractAndUpdateUserProfileMemory(
        cid
    )
    assert "profession" not in (result.get("profile") or {})
    assert result["facts"] == []


def test_atlas_profile_fields_include_required_key_value():
    from wrappers.mongo_store import atlas_profile_fields

    payload = atlas_profile_fields(
        {"profile": {"name": "Aayush"}, "facts": [], "memoryStatus": "COMPLETED"},
        "cid1",
    )
    assert payload["key"] == "profile"
    assert payload["value"]["profile"] == {"name": "Aayush"}
    assert payload["sourceConversationId"] == "cid1"
    assert payload["memoryStatus"] == "COMPLETED"
