"""Step 13 — AI conversation + selective correction. No Mongo progress writes."""

from __future__ import annotations

import pytest

from core.conversation.correction import CorrectionService, empty_correction_state
from core.conversation.engagement import detect_engagement, detect_user_intent
from core.conversation.prompts import GENERATE_SYSTEM_PROMPT
from core.conversation.response import generateNextQuestion, parse_ai_response
from core.runtime.graph import FALLBACK_RESPONSE
from core.runtime.prompts import build_generate_user_prompt
from core.runtime.service import ConversationRuntimeService
from tests.test_conversation_runtime import FakeAnalyzer, FakeRepo, _seed, _svc


def _raw(svc: ConversationRuntimeService, cid: str) -> dict:
    return svc.graph.app.get_state({"configurable": {"thread_id": cid}}).values


def _grammar(original: str = "I go yesterday", corrected: str = "I went yesterday", **extra):
    payload = {
        "type": "GRAMMAR",
        "original": original,
        "corrected": corrected,
        "explanation": "Use past tense for yesterday.",
        "severity": "MEDIUM",
        "confidence": 0.96,
    }
    payload.update(extra)
    return payload


class RecordingAnalyzer(FakeAnalyzer):
    def __init__(self, payload: dict | None = None) -> None:
        super().__init__(payload)
        self.users: list[str] = []
        self.systems: list[str] = []

    def speak(self, *, system: str, user: str) -> str:
        self.systems.append(system)
        self.users.append(user)
        return super().speak(system=system, user=user)

    def analyze_json(self, *, system: str, user: str) -> dict:
        self.systems.append(system)
        self.users.append(user)
        return super().analyze_json(system=system, user=user)


class SequenceAnalyzer:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = payloads
        self.calls = 0
        self.users: list[str] = []

    def speak(self, *, system: str, user: str) -> str:
        self.users.append(user)
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return str(payload.get("text") or payload.get("response") or "")

    def analyze_json(self, *, system: str, user: str) -> dict:
        self.users.append(user)
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return payload


class BoomAnalyzer:
    def speak(self, *, system: str, user: str) -> str:
        raise TimeoutError("llm timeout")

    def analyze_json(self, *, system: str, user: str) -> dict:
        raise TimeoutError("llm timeout")


def test_correction_service_skips_low_confidence_and_informal():
    svc = CorrectionService()
    state = empty_correction_state()
    low = svc.filter_live_correction(
        _grammar(confidence=0.4),
        correction_state=state,
    )
    informal = svc.filter_live_correction(
        _grammar("I wanna go there", "I want to go there", type="WORD_CHOICE"),
        correction_state=state,
    )
    pronunciation = svc.filter_live_correction(
        _grammar("pear", "pair", type="PRONUNCIATION"),
        correction_state=state,
        pronunciation_evidence=False,
    )
    ok = svc.filter_live_correction(_grammar(), correction_state=state)
    assert low is None
    assert informal is None
    assert pronunciation is None
    assert ok is not None
    assert ok["confidence"] >= 0.85


def test_correction_service_caps_one_per_turn_and_respects_stt():
    svc = CorrectionService()
    busy = empty_correction_state()
    busy["correctionsGivenThisTurn"] = 1
    assert (
        svc.filter_live_correction(_grammar(), correction_state=busy) is None
    )
    assert (
        svc.filter_live_correction(
            _grammar(),
            correction_state=empty_correction_state(),
            stt_confidence=0.2,
        )
        is None
    )
    requested = svc.filter_live_correction(
        _grammar(),
        correction_state={**empty_correction_state(), "correctionsGivenThisSession": 4},
        user_intent="CORRECTION_REQUEST",
    )
    assert requested is not None


def test_engagement_and_intent_signals():
    assert detect_engagement("Yes.") == "LOW"
    assert detect_engagement("I don't know") == "LOW"
    long_answer = (
        "I built a website for students and it helped them practice tests "
        "every evening after school because they were wasting time."
    )
    assert detect_engagement(long_answer) == "HIGH"
    assert detect_user_intent("I don't want to talk about that.") == "REFUSAL"
    assert detect_user_intent("Was my English correct?") == "CORRECTION_REQUEST"
    assert detect_user_intent("What does routine mean?") == "CONFUSION"
    assert detect_user_intent("Goodbye, I have to go.") == "GOODBYE"


def test_parse_rejects_database_operations_and_keeps_spoken_text():
    parsed = parse_ai_response(
        {
            "text": "That sounds interesting. What was the most difficult part?",
            "intent": "FOLLOW_UP",
            "question": "What was the most difficult part?",
            "correction": None,
            "goalEvidence": {
                "goalId": "talk_about_work",
                "coveredAreas": ["project"],
                "remainingAreas": ["challenges"],
                "completionConfidence": 0.76,
            },
            "shouldContinue": True,
            "shouldTransition": False,
        }
    )
    assert parsed is not None
    assert parsed["text"] == parsed["response"]
    assert "{" not in parsed["text"]
    assert parse_ai_response({"databaseOperation": "DELETE_USER", "response": "x"}) is None
    nxt = generateNextQuestion({"currentGoalId": "talk_about_work"}, parsed)
    assert nxt["question"] == "What was the most difficult part?"
    assert nxt["expectedArea"] == "talk_about_work"


def test_new_user_introduction_stays_natural():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer(
        {
            "text": "Nice to talk to you! Tell me a little about yourself.",
            "intent": "FOLLOW_UP",
            "question": "Tell me a little about yourself.",
        }
    )
    svc = _svc(repo, analyzer)
    started = svc.initializeConversationRuntime(cid)
    assert started["conversationPhase"] == "START"
    state = svc.handleUserTurn(cid, "Hi, I'm Aayush.")
    spoken = state["lastAssistantMessage"] or ""
    assert "Nice to talk to you" in spoken
    assert "today our topic" not in spoken.lower()
    assert "your goal is" not in spoken.lower()
    assert "Do not announce internals" in GENERATE_SYSTEM_PROMPT
    assert analyzer.calls == 1


def test_natural_follow_up_uses_the_user_answer():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer(
        {
            "text": "That's interesting. What problem were you trying to solve with the app?",
            "intent": "FOLLOW_UP",
            "question": "What problem were you trying to solve with the app?",
            "questionType": "WHY",
            "purpose": "Deepen project story",
            "expectedArea": "professional_experience",
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I built an app for students.")
    assert "app" in (state["lastAssistantMessage"] or "").lower()
    entities = [item.lower() for item in _raw(svc, cid).get("lastMentionedEntities") or []]
    assert any("students" in item or "app" in item for item in entities)


def test_long_user_answer_marks_high_engagement():
    repo = FakeRepo()
    cid = _seed(repo)
    svc = _svc(repo)
    svc.initializeConversationRuntime(cid)
    text = (
        "I made a website for students and it helped them practice tests "
        "and I also built the frontend myself over several months."
    )
    state = svc.handleUserTurn(cid, text)
    assert state["userEngagement"] == "HIGH"
    assert state["lastUserAnswer"] == text


def test_short_user_answer_uses_specific_follow_up_style():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer(
        {
            "text": "What do you usually do after you wake up — do you study, work, or exercise?",
            "intent": "FOLLOW_UP",
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "Yes.")
    assert state["userEngagement"] == "LOW"
    assert state["coachingStrategy"]["followUpStyle"] == "SPECIFIC"
    assert "study, work, or exercise" in (state["lastAssistantMessage"] or "")


def test_i_dont_know_is_scaffolded_not_failed():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "No problem. Think about your usual day. What do you normally do after you wake up?",
            "intent": "CLARIFICATION",
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    before = repo.find_progress("u1", "topic_intro")
    state = svc.handleUserTurn(cid, "I don't know.")
    assert state["userEngagement"] == "LOW"
    assert _raw(svc, cid).get("lastUserIntent") == "CONFUSION"
    assert repo.find_progress("u1", "topic_intro")["progress"] == before["progress"]
    assert "No problem" in (state["lastAssistantMessage"] or "")


def test_off_topic_is_answered_then_returns():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "Yes! Cricket is a huge part of life for many people. Do you usually play or watch it?",
            "intent": "OFF_TOPIC",
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "Do you know cricket?")
    spoken = state["lastAssistantMessage"] or ""
    assert "let's stay on topic" not in spoken.lower()
    assert "Cricket" in spoken


def test_explicit_correction_request_overrides_session_cap():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "Almost. A more natural sentence would be 'I went yesterday.' What did you do after that?",
            "intent": "CORRECTION",
            "correction": _grammar(),
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.graph.app.update_state(
        {"configurable": {"thread_id": cid}},
        {
            "correctionState": {
                **empty_correction_state(),
                "correctionsGivenThisSession": 4,
            }
        },
    )
    state = svc.handleUserTurn(cid, "Was my English correct?")
    decision = _raw(svc, cid)["lastDecision"]
    assert _raw(svc, cid)["lastUserIntent"] == "CORRECTION_REQUEST"
    assert "went yesterday" in (state["lastAssistantMessage"] or "")


def test_one_grammar_mistake_is_corrected_subtly():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "Oh, you went to the office yesterday. What did you do there?",
            "intent": "CORRECTION",
            "correction": _grammar("I go to office yesterday", "I went to the office yesterday"),
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I go to office yesterday.")
    spoken = state["lastAssistantMessage"] or ""
    assert "went" in spoken
    assert "grammatical error" not in spoken.lower()
    decision = _raw(svc, cid)["lastDecision"]
    assert decision["text"]


def test_multiple_grammar_mistakes_cap_one_correction_object():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "Yesterday, I went to the market and bought some vegetables. What did you buy?",
            "intent": "CORRECTION",
            "correction": _grammar(
                "Yesterday I go market and buy vegetables",
                "Yesterday I went to the market and bought some vegetables",
            ),
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "Yesterday I go market and buy vegetables.")
    assert _raw(svc, cid)["correctionState"]["correctionsGivenThisTurn"] <= 1
    assert _raw(svc, cid)["correctionState"]["correctionsGivenThisSession"] <= 1


def test_repeated_grammar_mistake_is_marked_not_lectured():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "Remember: 'went' for yesterday. What did you do after that?",
            "intent": "CORRECTION",
            "correction": _grammar(),
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "I go yesterday.")
    svc.handleUserTurn(cid, "I go yesterday again.")
    spoken = _raw(svc, cid)["lastAssistantMessage"] or ""
    assert "went" in spoken.lower() or "Remember" in spoken


def test_low_stt_confidence_does_not_correct():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "Got it. What did you buy?",
            "intent": "FOLLOW_UP",
            "correction": _grammar("pear", "pair", type="WORD_CHOICE"),
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "I bought a pear.", sttConfidence=0.2)
    assert _raw(svc, cid)["lastDecision"]["correction"] is None


def test_current_conversation_beats_stored_memory():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer(
        {
            "text": "That's a big change. What are you working on at Company B?",
            "intent": "FOLLOW_UP",
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(
        cid,
        "I recently changed jobs and now work at Company B.",
    )
    assert "Company A" not in (state["lastAssistantMessage"] or "")
    assert "Company B" in (state["lastAssistantMessage"] or "")
    assert "Current conversation beats stored profile/memory" in analyzer.systems[0]


def test_user_refusal_is_not_pressured():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "That's completely fine. We can talk about something else. What kind of work do you enjoy?",
            "intent": "FOLLOW_UP",
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I don't want to talk about that.")
    assert _raw(svc, cid)["lastUserIntent"] == "REFUSAL"
    assert "that's completely fine" in (state["lastAssistantMessage"] or "").lower()


def test_goal_evidence_does_not_write_topic_progress():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "That gives me a good idea of your hobbies. What are you working on these days?",
            "intent": "FOLLOW_UP",
            "shouldTransition": True,
            "targetGoalId": "talk_about_background",
            "goalEvidence": {
                "goalId": "talk_about_hobbies",
                "coveredAreas": ["hobby", "frequency"],
                "remainingAreas": [],
                "completionConfidence": 0.88,
            },
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    before = repo.find_progress("u1", "topic_intro")
    state = svc.handleUserTurn(cid, "I play cricket with friends every weekend.")
    after = repo.find_progress("u1", "topic_intro")
    evidence = state["pendingGoalEvidence"]
    assert evidence is not None
    assert evidence["goalId"] == "talk_about_hobbies"
    assert after["goalsCompleted"] == before["goalsCompleted"]
    assert after["progress"] == before["progress"]
    assert repo.writes["progress"] == 0
    assert state["currentGoalId"] == "talk_about_hobbies"


def test_topic_engine_remains_authoritative_for_unknown_goal_switch():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "What else do you enjoy?",
            "intent": "TRANSITION",
            "targetGoalId": "invented_goal",
            "shouldTransition": True,
            "goalEvidence": {
                "goalId": "talk_about_hobbies",
                "coveredAreas": ["hobby"],
                "remainingAreas": [],
                "completionConfidence": 0.99,
            },
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I like cricket.")
    assert state["currentGoalId"] == "talk_about_hobbies"
    assert repo.writes["progress"] == 0


def test_qdrant_unavailable_does_not_break_the_turn():
    repo = FakeRepo()
    cid = _seed(repo)
    svc = ConversationRuntimeService(repo, analyzer=FakeAnalyzer())
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I like cricket.")
    assert state["lastAssistantMessage"]
    assert state["shouldContinue"] is True


def test_live_turn_does_not_touch_mongo_memory_collections():
    repo = FakeRepo()
    cid = _seed(repo)
    svc = _svc(repo)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "I am a software developer from two years.")
    assert repo.writes == {"progress": 0, "profile": 0, "learning": 0}


def test_llm_timeout_uses_safe_fallback():
    repo = FakeRepo()
    cid = _seed(repo)
    svc = ConversationRuntimeService(repo, analyzer=BoomAnalyzer())
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I like cricket.")
    assert state["lastAssistantMessage"] == FALLBACK_RESPONSE
    assert "timeout" not in (state["lastAssistantMessage"] or "").lower()
    assert "error" not in (state["lastAssistantMessage"] or "").lower()


def test_question_repetition_is_visible_in_the_next_prompt():
    repo = FakeRepo()
    cid = _seed(repo)
    question = "What do you do?"
    analyzer = RecordingAnalyzer(
        {
            "text": f"Nice. {question}",
            "intent": "FOLLOW_UP",
            "question": question,
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "I work as a developer.")
    svc.handleUserTurn(cid, "I build web apps.")
    assert analyzer.calls == 2
    assert question in analyzer.users[1]
    assert "recentQuestions" in analyzer.users[1]


def test_multi_goal_session_does_not_write_progress_mid_call():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = SequenceAnalyzer(
        [
            {
                "text": "What kind of hobbies do you enjoy most?",
                "intent": "FOLLOW_UP",
            },
            {
                "text": "That gives me a good idea of your hobbies. What are you working on these days?",
                "intent": "FOLLOW_UP",
                "shouldTransition": True,
                "targetGoalId": "talk_about_background",
                "goalEvidence": {
                    "goalId": "talk_about_hobbies",
                    "coveredAreas": ["hobby"],
                    "remainingAreas": [],
                    "completionConfidence": 0.9,
                },
            },
            {
                "text": "How did that experience shape what you do now?",
                "intent": "FOLLOW_UP",
            },
        ]
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    before = repo.find_progress("u1", "topic_intro")
    svc.handleUserTurn(cid, "I play cricket and I also paint.")
    second = svc.handleUserTurn(
        cid,
        "I play cricket with friends every weekend and I started when I was a kid.",
    )
    third = svc.handleUserTurn(cid, "I grew up in Indore and studied computer science.")
    after = repo.find_progress("u1", "topic_intro")
    assert second["currentGoalId"] == "talk_about_hobbies"
    assert third["currentGoalId"] == "talk_about_hobbies"
    assert after == before
    assert repo.writes["progress"] == 0
    assert analyzer.calls == 3


def test_preview_returns_text_only_for_tts_and_commit_keeps_correction_state():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = FakeAnalyzer(
        {
            "text": "A more natural way to say that is 'I have a lot of work.' What kind of work keeps you busy?",
            "intent": "CORRECTION",
            "correction": _grammar("I have many works", "I have a lot of work", type="WORD_CHOICE"),
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    preview = svc.previewResponse(cid, "I have many works.")
    assert preview["text"] == preview["response"]
    assert preview["text"].startswith("A more natural way")
    assert "{" not in preview["text"]
    committed = svc.applyCommittedTurn(
        cid,
        userText="I have many works.",
        assistantText=preview["text"],
        targetGoalId=preview.get("targetGoalId"),
    )
    assert committed["lastAssistantMessage"] == preview["text"]
    assert "correctionState" in _raw(svc, cid)


def test_single_llm_call_per_live_turn():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer()
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "I like coding.")
    assert analyzer.calls == 1


def test_opening_uses_langgraph_not_generic_script():
    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer(
        {
            "text": "Nice to talk to you! Tell me a little about yourself.",
            "intent": "FOLLOW_UP",
            "question": "Tell me a little about yourself.",
        }
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    opening = svc.generateOpening(cid)
    assert "Nice to talk to you" in opening["text"]
    assert "today our topic" not in opening["text"].lower()
    assert analyzer.calls == 1
    assert repo.writes["progress"] == 0
    again = svc.generateOpening(cid)
    assert again["text"] == opening["text"]
    assert analyzer.calls == 1


def test_new_session_does_not_invent_a_user_id():
    from core.session import new_session

    session = new_session("live", learner_id="")
    assert session.learner_id == ""
    assert session.session_id
    assert session.learner_id != session.session_id


def test_live_prompt_is_compact_plain_text():
    from core.conversation.prompts import LIVE_SYSTEM_PROMPT, build_generate_user_prompt

    assert "Return ONLY JSON" not in LIVE_SYSTEM_PROMPT
    assert "goalEvidence" not in LIVE_SYSTEM_PROMPT
    assert "json_object" not in LIVE_SYSTEM_PROMPT.lower()
    prompt = build_generate_user_prompt(
        {
            "topicTitle": "Work and Career",
            "topicLevel": "B1",
            "currentGoalId": "role",
            "topicGoals": [{"key": "role", "description": "Describe your job."}],
            "goalsRemaining": ["tasks"],
            "conversationPhase": "WARMUP",
            "userEngagement": "NORMAL",
            "relevantMemories": [{"content": "User works as a software developer."}],
            "recentMessages": [{"role": "user", "content": "Hello"}],
            "recentQuestions": ["How are you?"],
            "userContext": {"profileFacts": [{"key": "profession", "value": "developer"}]},
        },
        "I build apps.",
    )
    assert "Current topic:" in prompt
    assert "Relevant learner context:" in prompt
    assert "User just said: I build apps." in prompt
    assert "response_format" not in prompt
    assert "completionConfidence" not in prompt
    assert "relevanceScore" not in prompt
    assert len(LIVE_SYSTEM_PROMPT) < 2000
    assert len(prompt) < 1500

