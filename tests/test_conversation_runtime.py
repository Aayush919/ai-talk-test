"""LangGraph runtime state — short-term conversational brain, not Mongo."""

from __future__ import annotations

from uuid import uuid4

import pytest

from core.conversations.errors import ConversationNotActive, ConversationNotFound
from core.runtime.config import RUNTIME_CONFIG
from core.runtime.service import ConversationRuntimeService


def _topic() -> dict:
    return {
        "_id": "topic_intro",
        "title": "Introduction",
        "slug": "a1-introduction",
        "level": "A1",
        "goals": [
            {"key": "introduce_self", "description": "User can introduce their name."},
            {"key": "talk_about_work", "description": "User can talk about work."},
            {"key": "talk_about_hobbies", "description": "User can talk about hobbies."},
            {"key": "talk_about_background", "description": "User can talk about background."},
        ],
    }


class FakeAnalyzer:
    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {
            "response": "What do you usually enjoy doing in your free time?",
            "intent": "FOLLOW_UP",
            "followUpNeeded": True,
        }
        self.calls = 0
        self.last_system = ""
        self.last_user = ""

    def speak(self, *, system: str, user: str) -> str:
        self.calls += 1
        self.last_system = system
        self.last_user = user
        data = self.payload or {}
        if any(
            key in data
            for key in (
                "databaseOperation",
                "topicProgress",
                "userId",
                "conversationId",
                "goalsCompleted",
                "goalsRemaining",
            )
        ):
            return ""
        return str(data.get("text") or data.get("response") or "")

    def analyze_json(self, *, system: str, user: str) -> dict:
        self.calls += 1
        return self.payload


class FakeRepo:
    def __init__(self) -> None:
        self.sessions: list[dict] = []
        self.topics: list[dict] = [_topic()]
        self.progress: list[dict] = []
        self.profiles: list[dict] = []
        self.learning: list[dict] = []
        self.messages: list[dict] = []
        self.writes = {"progress": 0, "profile": 0, "learning": 0}

    def find_conversation_session(self, conversation_id: str) -> dict | None:
        for row in self.sessions:
            if str(row["_id"]) == str(conversation_id):
                return dict(row)
        return None

    def find_topic(self, topic_id) -> dict | None:
        for topic in self.topics:
            if topic["_id"] == topic_id or str(topic["_id"]) == str(topic_id):
                return dict(topic)
        return None

    def find_progress(self, user_id: str, topic_id) -> dict | None:
        for row in self.progress:
            if row["userId"] == user_id and str(row["topicId"]) == str(topic_id):
                return dict(row)
        return None

    def find_user_profile(self, user_id: str) -> dict | None:
        for row in self.profiles:
            if str(row["userId"]) == str(user_id):
                return dict(row)
        return None

    def find_learning_memory(self, user_id: str) -> dict | None:
        for row in self.learning:
            if str(row["userId"]) == str(user_id):
                return dict(row)
        return None

    def list_progress(self, user_id: str) -> list[dict]:
        return [row for row in self.progress if row["userId"] == user_id]

    def list_active_topics(self, level: str) -> list[dict]:
        rows = [
            topic
            for topic in self.topics
            if topic.get("isActive", True)
            and (not level or not topic.get("level") or topic.get("level") == level)
        ]
        rows.sort(key=lambda item: int(item.get("order") or 0))
        return rows

    def list_messages(self, conversation_id: str) -> list[dict]:
        raise AssertionError("runtime must not load lifetime messages")

    def apply_progress_from_conversation(self, *args, **kwargs):
        self.writes["progress"] += 1
        raise AssertionError("runtime must not write topic_progress")

    def apply_profile_from_conversation(self, *args, **kwargs):
        self.writes["profile"] += 1
        raise AssertionError("runtime must not write user_profile_memory")

    def apply_learning_memory_from_conversation(self, *args, **kwargs):
        self.writes["learning"] += 1
        raise AssertionError("runtime must not write learning_memory")


def _seed(
    repo: FakeRepo,
    *,
    status: str = "ACTIVE",
    topic_id: str = "topic_intro",
    user_id: str = "u1",
    progress: int = 40,
) -> str:
    cid = uuid4().hex
    repo.sessions.append(
        {
            "_id": cid,
            "userId": user_id,
            "topicId": topic_id,
            "status": status,
        }
    )
    repo.progress.append(
        {
            "userId": user_id,
            "topicId": topic_id,
            "status": "IN_PROGRESS",
            "progress": progress,
            "goalsCompleted": ["introduce_self", "talk_about_work"],
            "goalsRemaining": ["talk_about_hobbies", "talk_about_background"],
        }
    )
    repo.profiles.append(
        {
            "userId": user_id,
            "profile": {
                "name": "Aayush",
                "profession": "software developer",
                "hobbies": ["cricket"],
                "nativeLanguage": "Hindi",
            },
            "facts": [],
        }
    )
    repo.learning.append(
        {
            "userId": user_id,
            "recurringMistakes": [
                {
                    "category": "grammar",
                    "skill": "past_tense",
                    "status": "ACTIVE",
                    "frequency": 4,
                }
            ],
        }
    )
    for index in range(40):
        repo.messages.append(
            {
                "conversationId": "lifetime",
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"old message {index}",
                "sequence": index,
            }
        )
    return cid


def _svc(repo: FakeRepo, analyzer: FakeAnalyzer | None = None) -> ConversationRuntimeService:
    return ConversationRuntimeService(repo, analyzer=analyzer or FakeAnalyzer())


def test_runtime_initialization_builds_valid_state():
    repo = FakeRepo()
    cid = _seed(repo)
    state = _svc(repo).initializeConversationRuntime(cid)
    assert state["conversationId"] == cid
    assert state["userId"] == "u1"
    assert state["topicTitle"] == "Introduction"
    assert state["shouldContinue"] is True
    assert state["conversationTurn"] == 0
    assert state["recentMessages"] == []
    assert state["coachingStrategy"]["correctionMode"] == "SUBTLE"


def test_missing_session_raises_not_found():
    with pytest.raises(ConversationNotFound) as err:
        _svc(FakeRepo()).initializeConversationRuntime("missing")
    assert err.value.code == "CONVERSATION_NOT_FOUND"


def test_completed_session_cannot_start_runtime():
    repo = FakeRepo()
    cid = _seed(repo, status="COMPLETED")
    with pytest.raises(ConversationNotActive) as err:
        _svc(repo).initializeConversationRuntime(cid)
    assert err.value.code == "CONVERSATION_NOT_ACTIVE"


def test_topic_comes_from_conversation_session_not_frontend():
    repo = FakeRepo()
    cid = _seed(repo, topic_id="topic_intro")
    state = _svc(repo).initializeConversationRuntime(cid)
    assert state["topicId"] == "topic_intro"
    assert state["currentGoalId"] == "talk_about_hobbies"
    assert "introduce_self" in state["goalsCompleted"]


def test_recent_messages_stay_within_limit():
    repo = FakeRepo()
    cid = _seed(repo)
    svc = _svc(repo)
    svc.initializeConversationRuntime(cid)
    for index in range(10):
        svc.handleUserTurn(cid, f"I like playing cricket {index}")
    state = svc.getRuntimeState(cid)
    assert len(state["recentMessages"]) <= RUNTIME_CONFIG["maxRecentMessages"]
    assert len(state["recentMessages"]) == 20


def test_each_user_turn_increments_conversation_turn():
    repo = FakeRepo()
    cid = _seed(repo)
    svc = _svc(repo)
    svc.initializeConversationRuntime(cid)
    first = svc.handleUserTurn(cid, "My name is Aayush.")
    second = svc.handleUserTurn(cid, "I am a software developer.")
    assert first["conversationTurn"] == 1
    assert second["conversationTurn"] == 2
    assert second["lastUserMessage"] == "I am a software developer."


def test_runtime_goal_can_switch_without_completing_database_goal():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "response": "Do you usually play with friends or family?",
            "intent": "TRANSITION",
            "targetGoalId": "talk_about_background",
            "followUpNeeded": True,
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    before = repo.find_progress("u1", "topic_intro")
    state = svc.handleUserTurn(cid, "I like playing cricket.")
    after = repo.find_progress("u1", "topic_intro")
    assert state["currentGoalId"] == "talk_about_background"
    assert after["goalsRemaining"] == before["goalsRemaining"]
    assert after["goalsCompleted"] == before["goalsCompleted"]
    assert after["progress"] == before["progress"]
    assert repo.writes["progress"] == 0


def test_relevant_profile_context_is_loaded():
    repo = FakeRepo()
    cid = _seed(repo)
    state = _svc(repo).initializeConversationRuntime(cid)
    facts = {row["key"]: row["value"] for row in state["userContext"]["profileFacts"]}
    assert facts["name"] == "Aayush"
    assert facts["profession"] == "software developer"
    assert "cricket" in facts.values()


def test_relevant_learning_signals_are_loaded():
    repo = FakeRepo()
    cid = _seed(repo)
    state = _svc(repo).initializeConversationRuntime(cid)
    skills = {row["skill"]: row["status"] for row in state["userContext"]["learningSignals"]}
    assert skills["past_tense"] == "ACTIVE"


def test_runtime_does_not_load_lifetime_message_history():
    repo = FakeRepo()
    cid = _seed(repo)
    state = _svc(repo).initializeConversationRuntime(cid)
    assert state["recentMessages"] == []
    contents = " ".join(item["content"] for item in state["recentMessages"])
    assert "old message" not in contents


def test_conversation_threads_are_isolated():
    repo = FakeRepo()
    first = _seed(repo, user_id="u1")
    second = _seed(repo, user_id="u1")
    svc = _svc(repo)
    svc.initializeConversationRuntime(first)
    svc.initializeConversationRuntime(second)
    svc.handleUserTurn(first, "I play cricket.")
    svc.handleUserTurn(second, "I wake up at seven.")
    a = svc.getRuntimeState(first)
    b = svc.getRuntimeState(second)
    assert a["lastUserMessage"] == "I play cricket."
    assert b["lastUserMessage"] == "I wake up at seven."
    assert a["conversationId"] != b["conversationId"]


def test_checkpoint_can_be_restored_by_conversation_id():
    repo = FakeRepo()
    cid = _seed(repo)
    first = _svc(repo)
    first.initializeConversationRuntime(cid)
    first.handleUserTurn(cid, "I like cricket.")
    restored = ConversationRuntimeService(
        repo,
        analyzer=FakeAnalyzer(),
        checkpointer=first.checkpointer,
    )
    state = restored.getRuntimeState(cid)
    assert state["conversationId"] == cid
    assert state["lastUserMessage"] == "I like cricket."
    assert state["conversationTurn"] == 1
    assert state["shouldContinue"] is True


def test_runtime_end_sets_should_continue_false():
    repo = FakeRepo()
    cid = _seed(repo)
    svc = _svc(repo)
    svc.initializeConversationRuntime(cid)
    ended = svc.endConversationRuntime(cid)
    assert ended["shouldContinue"] is False
    with pytest.raises(ConversationNotActive):
        svc.handleUserTurn(cid, "hello")


def test_runtime_does_not_mutate_permanent_memory():
    repo = FakeRepo()
    cid = _seed(repo)
    profile_before = repo.find_user_profile("u1")
    learning_before = repo.find_learning_memory("u1")
    progress_before = repo.find_progress("u1", "topic_intro")
    svc = _svc(repo)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "I am a software developer.")
    assert repo.find_user_profile("u1") == profile_before
    assert repo.find_learning_memory("u1") == learning_before
    assert repo.find_progress("u1", "topic_intro") == progress_before
    assert repo.writes == {"progress": 0, "profile": 0, "learning": 0}


def test_invalid_llm_decision_is_rejected_safely():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "databaseOperation": "DELETE_USER",
            "topicProgress": 100,
            "response": "Your current goal is goal_3.",
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    progress_before = repo.find_progress("u1", "topic_intro")
    state = svc.handleUserTurn(cid, "I like cricket.")
    assert state["lastAssistantMessage"] == "Nice. Tell me a bit more."
    assert "goal_3" not in (state["lastAssistantMessage"] or "")
    assert repo.find_progress("u1", "topic_intro")["progress"] == progress_before["progress"]
    assert repo.writes["progress"] == 0
