"""Learning memory — patterns, not one-off mistakes. Idempotent, isolated from profile/progress."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.conversations.errors import ConversationNotCompleted, SummaryNotFound
from core.memory.learning_config import LEARNING_MEMORY_CONFIG, LearningMemoryConfig
from core.memory.learning_service import LearningMemoryService
from core.topics.progress_service import TopicProgressService


def _now():
    return datetime.now(timezone.utc)


def _signal(
    category: str,
    skill: str,
    issue: str,
    *,
    severity: str = "medium",
    confidence: float = 0.91,
) -> dict:
    return {
        "category": category,
        "skill": skill,
        "issue": issue,
        "severity": severity,
        "confidence": confidence,
    }


def _payload(*, signals=None, strengths=None, patterns=None) -> dict:
    return {
        "signals": list(signals or []),
        "strengths": list(strengths or []),
        "patterns": list(patterns or []),
    }


PAST_TENSE = _signal(
    "grammar",
    "past_tense",
    "difficulty using past tense consistently",
)


class FakeAnalyzer:
    def __init__(self, payload: dict | None = None, payloads: list[dict] | None = None) -> None:
        self.payloads = list(payloads) if payloads is not None else [payload or _payload()]
        self.calls = 0

    def analyze_json(self, *, system: str, user: str) -> dict:
        idx = min(self.calls, len(self.payloads) - 1)
        self.calls += 1
        return self.payloads[idx]


class FakeRepo:
    def __init__(self) -> None:
        self.users: dict[str, dict] = {"u1": {"_id": "u1", "englishLevel": "A1"}}
        self.sessions: list[dict] = []
        self.summaries: list[dict] = []
        self.learning: list[dict] = []
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

    def find_learning_memory(self, user_id: str) -> dict | None:
        for row in self.learning:
            if str(row["userId"]) == str(user_id):
                return dict(row)
        return None

    def apply_learning_memory_from_conversation(
        self,
        user_id: str,
        conversation_id: str,
        fields: dict,
    ) -> dict | None:
        with self._lock:
            cid = str(conversation_id)
            for row in self.learning:
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
            self.learning.append(doc)
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
    status: str = "COMPLETED",
    summary: str | None = "The learner practiced past events.",
    summary_status: str = "COMPLETED",
    mistakes: list | None = None,
    goals: list | None = None,
    include_summary: bool = True,
) -> str:
    cid = uuid4().hex
    repo.sessions.append(
        {
            "_id": cid,
            "userId": user_id,
            "topicId": "t1",
            "status": status,
            "languageLevelAtStart": "A1",
        }
    )
    if include_summary:
        repo.summaries.append(
            {
                "conversationId": cid,
                "userId": user_id,
                "topicId": "t1",
                "summaryStatus": summary_status,
                "summary": summary,
                "mistakes": mistakes or [],
                "goals": goals
                or [
                    {"goalId": "g1", "status": "COMPLETED", "evidence": "ok"},
                    {"goalId": "g2", "status": "COMPLETED", "evidence": "ok"},
                    {"goalId": "g3", "status": "COMPLETED", "evidence": "ok"},
                    {"goalId": "g4", "status": "NOT_ATTEMPTED", "evidence": ""},
                    {"goalId": "g5", "status": "NOT_ATTEMPTED", "evidence": ""},
                ],
                "grammarPatterns": [],
                "fluencyObservations": [],
                "strengths": [],
                "weaknesses": [],
            }
        )
    return cid


def _svc(repo: FakeRepo, analyzer: FakeAnalyzer, **config) -> LearningMemoryService:
    mapping = dict(LEARNING_MEMORY_CONFIG)
    mapping.update(config)
    return LearningMemoryService(
        repo,
        analyzer=analyzer,
        config=LearningMemoryConfig.from_mapping(mapping),
    )


def _mistake(result: dict, skill: str = "past_tense") -> dict:
    for row in result["recurringMistakes"]:
        if row.get("skill") == skill:
            return row
    raise AssertionError(f"missing skill {skill}: {result['recurringMistakes']}")


def test_single_grammar_mistake_is_not_a_strong_recurring_weakness():
    repo = FakeRepo()
    cid = _seed(repo)
    result = _svc(repo, FakeAnalyzer(_payload(signals=[PAST_TENSE]))).analyzeAndUpdateLearningMemory(
        cid
    )
    row = _mistake(result)
    assert row["frequency"] == 1
    assert row["status"] == "ACTIVE"
    assert result["skills"]["grammar"][0]["frequency"] == 1
    assert result["improvementAreas"] == []


def test_same_pattern_across_three_conversations_is_active_recurring():
    repo = FakeRepo()
    analyzer = FakeAnalyzer(_payload(signals=[PAST_TENSE]))
    svc = _svc(repo, analyzer)
    last = None
    for _ in range(3):
        last = svc.analyzeAndUpdateLearningMemory(_seed(repo))
    row = _mistake(last)
    assert row["frequency"] == 3
    assert row["status"] == "ACTIVE"
    assert any(area["skill"] == "past_tense" for area in last["improvementAreas"])


def test_repeated_absence_marks_issue_improving():
    repo = FakeRepo()
    present = FakeAnalyzer(_payload(signals=[PAST_TENSE]))
    svc = _svc(repo, present)
    for _ in range(3):
        svc.analyzeAndUpdateLearningMemory(_seed(repo))
    empty = FakeAnalyzer(_payload())
    svc.analyzer = empty
    svc.analyzeAndUpdateLearningMemory(_seed(repo))
    result = svc.analyzeAndUpdateLearningMemory(_seed(repo))
    row = _mistake(result)
    assert row["frequency"] == 3
    assert row["status"] == "IMPROVING"


def test_sustained_absence_marks_issue_resolved():
    repo = FakeRepo()
    svc = _svc(repo, FakeAnalyzer(_payload(signals=[PAST_TENSE])))
    for _ in range(3):
        svc.analyzeAndUpdateLearningMemory(_seed(repo))
    svc.analyzer = FakeAnalyzer(_payload())
    last = None
    for _ in range(4):
        last = svc.analyzeAndUpdateLearningMemory(_seed(repo))
    row = _mistake(last)
    assert row["status"] == "RESOLVED"
    assert row["frequency"] == 3


def test_strength_is_stored():
    repo = FakeRepo()
    cid = _seed(repo, summary="The learner used everyday vocabulary well.")
    result = _svc(
        repo,
        FakeAnalyzer(
            _payload(
                strengths=[
                    {
                        "skill": "vocabulary",
                        "description": "Good everyday vocabulary",
                        "confidence": 0.89,
                    }
                ]
            )
        ),
    ).analyzeAndUpdateLearningMemory(cid)
    assert result["strengths"][0]["description"] == "Good everyday vocabulary"


def test_summary_mistakes_fill_learning_when_llm_is_empty():
    repo = FakeRepo()
    cid = _seed(
        repo,
        mistakes=[
            {
                "type": "GRAMMAR",
                "userText": "I am living in Chandigarh",
                "correction": "I live in Chandigarh",
                "explanation": "Use present simple for a home city.",
            }
        ],
    )
    repo.summaries[-1]["strengths"] = [
        "Clear statement of occupation using simple present."
    ]
    result = _svc(repo, FakeAnalyzer(_payload())).analyzeAndUpdateLearningMemory(cid)
    assert result["recurringMistakes"]
    assert "living" in result["recurringMistakes"][0]["issue"].lower()
    assert result["strengths"]


def test_pronunciation_without_evidence_is_not_stored():
    repo = FakeRepo()
    cid = _seed(
        repo,
        mistakes=[
            {
                "type": "GRAMMAR",
                "userText": "I go yesterday",
                "correction": "I went yesterday",
                "explanation": "past tense",
            }
        ],
    )
    result = _svc(
        repo,
        FakeAnalyzer(
            _payload(
                signals=[
                    _signal(
                        "pronunciation",
                        "th_sound",
                        "difficulty pronouncing th sound",
                        confidence=0.95,
                    )
                ]
            )
        ),
    ).analyzeAndUpdateLearningMemory(cid)
    assert result["recurringMistakes"] == []
    assert result["skills"]["pronunciation"] == []


def test_duplicate_processing_does_not_increment_frequency():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(_payload(signals=[PAST_TENSE]))
    svc = _svc(repo, analyzer)
    first = svc.analyzeAndUpdateLearningMemory(cid)
    second = svc.analyzeAndUpdateLearningMemory(cid)
    assert analyzer.calls == 1
    assert _mistake(first)["frequency"] == 1
    assert _mistake(second)["frequency"] == 1
    assert first["metadata"]["totalAnalyzedConversations"] == 1
    assert second["metadata"]["totalAnalyzedConversations"] == 1
    assert len(repo.learning) == 1


def test_failed_conversation_does_not_update_learning_memory():
    repo = FakeRepo()
    now = _now()
    repo.learning.append(
        {
            "userId": "u1",
            "skills": {"grammar": []},
            "recurringMistakes": [],
            "strengths": [],
            "improvementAreas": [],
            "learningPatterns": [],
            "overallAssessment": {},
            "metadata": {"totalAnalyzedConversations": 2},
            "processedConversationIds": ["old"],
            "version": 1,
            "createdAt": now,
            "updatedAt": now,
        }
    )
    cid = _seed(repo, status="FAILED")
    analyzer = FakeAnalyzer(_payload(signals=[PAST_TENSE]))
    with pytest.raises(ConversationNotCompleted):
        _svc(repo, analyzer).analyzeAndUpdateLearningMemory(cid)
    assert analyzer.calls == 0
    assert repo.learning[0]["recurringMistakes"] == []
    assert repo.learning[0]["metadata"]["totalAnalyzedConversations"] == 2
    assert repo.learning[0]["version"] == 1


def test_missing_summary_raises():
    repo = FakeRepo()
    cid = _seed(repo, include_summary=False)
    with pytest.raises(SummaryNotFound) as err:
        _svc(repo, FakeAnalyzer(_payload(signals=[PAST_TENSE]))).analyzeAndUpdateLearningMemory(
            cid
        )
    assert err.value.code == "SUMMARY_NOT_FOUND"
    assert repo.learning == []


def test_invalid_category_is_ignored():
    repo = FakeRepo()
    cid = _seed(repo)
    result = _svc(
        repo,
        FakeAnalyzer(
            _payload(
                signals=[
                    _signal("favorite_car", "bmw", "likes BMW", confidence=0.99)
                ]
            )
        ),
    ).analyzeAndUpdateLearningMemory(cid)
    assert result["recurringMistakes"] == []


def test_low_confidence_signal_is_ignored():
    repo = FakeRepo()
    cid = _seed(repo)
    result = _svc(
        repo,
        FakeAnalyzer(
            _payload(
                signals=[
                    _signal(
                        "grammar",
                        "past_tense",
                        "maybe past tense",
                        confidence=0.45,
                    )
                ]
            )
        ),
    ).analyzeAndUpdateLearningMemory(cid)
    assert result["recurringMistakes"] == []


def test_topic_progress_is_not_copied_into_learning_memory():
    repo = FakeRepo()
    cid = _seed(
        repo,
        summary="The learner completed 3 of 5 Introduction goals.",
        goals=[
            {"goalId": "g1", "status": "COMPLETED", "evidence": "ok"},
            {"goalId": "g2", "status": "COMPLETED", "evidence": "ok"},
            {"goalId": "g3", "status": "COMPLETED", "evidence": "ok"},
            {"goalId": "g4", "status": "NOT_ATTEMPTED", "evidence": ""},
            {"goalId": "g5", "status": "NOT_ATTEMPTED", "evidence": ""},
        ],
    )
    progress = TopicProgressService(repo).updateTopicProgressFromSummary(cid)
    result = _svc(
        repo,
        FakeAnalyzer(
            _payload(
                signals=[
                    _signal("grammar", "topic_progress", "3/5 goals completed", confidence=0.99)
                ]
            )
        ),
    ).analyzeAndUpdateLearningMemory(cid)
    assert progress["progress"] == 60
    blob = str(result).lower()
    assert "3/5" not in blob
    assert "goals completed" not in blob
    assert result["recurringMistakes"] == []


def test_profile_facts_are_not_stored_as_learning_weaknesses():
    repo = FakeRepo()
    cid = _seed(repo, summary="The user said they are a software developer.")
    result = _svc(
        repo,
        FakeAnalyzer(
            _payload(
                signals=[
                    _signal(
                        "grammar",
                        "profession",
                        "software developer",
                        confidence=0.99,
                    )
                ]
            )
        ),
    ).analyzeAndUpdateLearningMemory(cid)
    blob = str(result).lower()
    assert "software developer" not in blob
    assert result["recurringMistakes"] == []


def test_source_conversation_ids_are_deduplicated_and_frequency_aggregates():
    repo = FakeRepo()
    analyzer = FakeAnalyzer(_payload(signals=[PAST_TENSE]))
    svc = _svc(repo, analyzer)
    first = _seed(repo)
    svc.analyzeAndUpdateLearningMemory(first)
    svc.analyzeAndUpdateLearningMemory(first)
    second = _seed(repo)
    third = _seed(repo)
    result = svc.analyzeAndUpdateLearningMemory(second)
    result = svc.analyzeAndUpdateLearningMemory(third)
    row = _mistake(result)
    assert row["frequency"] == 3
    assert row["sourceConversationIds"] == [first, second, third]
    assert len(set(row["sourceConversationIds"])) == 3


def test_document_growth_limits_are_respected():
    repo = FakeRepo()
    cid = _seed(repo)
    signals = [
        _signal("grammar", f"skill_{index}", f"issue {index}", confidence=0.9)
        for index in range(4)
    ]
    strengths = [
        {"skill": "vocabulary", "description": f"strength {index}", "confidence": 0.9}
        for index in range(4)
    ]
    patterns = [
        {"description": f"pattern {index}", "confidence": 0.9} for index in range(4)
    ]
    result = _svc(
        repo,
        FakeAnalyzer(_payload(signals=signals, strengths=strengths, patterns=patterns)),
        maxRecurringMistakes=2,
        maxStrengths=2,
        maxPatterns=2,
    ).analyzeAndUpdateLearningMemory(cid)
    assert len(result["recurringMistakes"]) == 2
    assert len(result["strengths"]) == 2
    assert len(result["learningPatterns"]) == 2
    assert LEARNING_MEMORY_CONFIG["maxRecurringMistakes"] == 50
    assert LEARNING_MEMORY_CONFIG["maxStrengths"] == 30
    assert LEARNING_MEMORY_CONFIG["maxPatterns"] == 30


def test_atlas_learning_fields_include_required_type_string_value():
    from wrappers.mongo_store import atlas_learning_fields

    payload = atlas_learning_fields(
        {
            "skills": {"grammar": []},
            "overallAssessment": {"level": "A1"},
            "updatedAt": _now(),
        },
        "cid1",
    )
    assert payload["type"] == "learning"
    assert payload["value"] == "A1"
    assert isinstance(payload["value"], str)
    assert payload["sourceConversationId"] == "cid1"
    assert payload["lastObservedAt"] == payload["updatedAt"]
