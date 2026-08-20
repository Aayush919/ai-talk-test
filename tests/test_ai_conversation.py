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
    assert detect_user_intent("Do you know what I am doing in my evening?") == "MEMORY_PROBE"
    assert detect_user_intent("You know my morning routine?") == "MEMORY_PROBE"
    assert detect_user_intent("Why are you again and again asking?") == "REPEAT_COMPLAINT"
    assert detect_user_intent("I already told you I am doing") == "REPEAT_COMPLAINT"
    assert detect_user_intent("You already asked me that.") == "REPEAT_COMPLAINT"
    assert detect_user_intent("What happened? I don't get your point.") == "CONFUSION"
    assert detect_user_intent("Yeah. I don't understand.") == "CONFUSION"


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
    assert evidence["goalId"] == "talk_about_background"
    assert after["goalsCompleted"] == before["goalsCompleted"]
    assert after["progress"] == before["progress"]
    assert repo.writes["progress"] == 0
    assert state["currentGoalId"] == "talk_about_background"


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
    assert state["currentGoalId"] == "talk_about_background"
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
    assert "Already asked (do not repeat):" in analyzer.users[1]


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
    assert second["currentGoalId"] == "talk_about_background"
    assert third["currentGoalId"] == "talk_about_background"
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
    assert "Local now (India):" in prompt
    assert "Do not ask what day or time it is" in prompt
    assert "Already known (do not re-ask):" in prompt
    assert "If they ask whether you know or remember" in LIVE_SYSTEM_PROMPT
    assert "profession: developer" in prompt
    assert "Relevant learner context:" in prompt
    assert "User just said: I build apps." in prompt
    assert "response_format" not in prompt
    assert "completionConfidence" not in prompt
    assert "relevanceScore" not in prompt
    assert len(LIVE_SYSTEM_PROMPT) < 2400
    assert len(prompt) < 1800


def test_live_facts_are_remembered_without_mongo_writes():
    from core.conversation.live_facts import extract_live_facts

    facts = extract_live_facts("I am Ayush Malviya. I am living in Bhopal. I am doing BTech.")
    keys = {row["key"] for row in facts}
    assert "name" in keys
    assert "location" in keys
    assert "education" in keys

    repo = FakeRepo()
    cid = _seed(
        repo,
        progress=0,
    )
    repo.progress[0]["goalsCompleted"] = []
    repo.progress[0]["goalsRemaining"] = [
        "introduce_self",
        "talk_about_work",
        "talk_about_hobbies",
        "talk_about_background",
    ]
    analyzer = RecordingAnalyzer({"text": "Nice to meet you, Ayush. Where do you live?"})
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I am Ayush Malviya. I live in Bhopal.")
    facts_state = (state.get("userContext") or {}).get("profileFacts") or []
    assert any(row.get("key") == "name" and "Ayush" in str(row.get("value")) for row in facts_state)
    assert "introduce_self" in (state.get("goalsCompleted") or [])
    assert state["currentGoalId"] != "introduce_self"
    prompt = analyzer.users[-1]
    assert "Already known (do not re-ask):" in prompt
    assert "Ayush" in prompt
    assert repo.writes["progress"] == 0
    assert repo.find_progress("u1", "topic_intro")["goalsCompleted"] == []


def test_spoken_hobbies_are_remembered_across_later_turns():
    from core.conversation.live_facts import extract_live_facts, merge_live_facts

    first = extract_live_facts(
        "Yeah. For fun, I'm watching movies. I'm playing cricket with friends."
    )
    hobby = next(row["value"].lower() for row in first if row["key"] == "hobby")
    assert "cricket" in hobby
    assert "movies" in hobby
    merged = merge_live_facts(first, extract_live_facts("I read business books."))
    hobby = next(row["value"].lower() for row in merged if row["key"] == "hobby")
    assert "reading books" in hobby
    assert "cricket" in hobby

    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer(
        {"text": "Nice. Do you remember I already know your hobbies."}
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(
        cid, "For fun I'm watching movies. I'm playing cricket with friends."
    )
    svc.handleUserTurn(cid, "I read business books.")
    svc.handleUserTurn(cid, "Do you remember all of my hobbies or not?")
    prompt = analyzer.users[-1]
    assert "Already known (do not re-ask):" in prompt
    assert "cricket" in prompt.lower()
    assert "movies" in prompt.lower()
    assert "If they ask whether you know or remember" in analyzer.systems[-1]
    assert repo.writes["progress"] == 0


def test_call_chat_context_keeps_early_answers_in_the_live_prompt():
    from core.conversation.chat_context import CallChatContext, MAX_CALL_CHARS
    from core.conversation.prompts import build_generate_user_prompt

    messages = []
    for index in range(12):
        messages.append(
            {
                "role": "assistant",
                "content": f"What time do you wake up on day {index}?",
            }
        )
        messages.append({"role": "user", "content": f"I wake up at 5:30 on day {index}."})
    prompt = build_generate_user_prompt(
        {
            "topicTitle": "Daily routine",
            "currentGoalId": "wake_up",
            "recentMessages": messages,
        },
        "Do you remember what time I wake up?",
    )
    assert "This call (do not re-ask anything already answered here):" in prompt
    assert "I wake up at 5:30 on day 0." in prompt
    assert "I wake up at 5:30 on day 11." in prompt
    assert "Covered this call:" in prompt

    long_ctx = CallChatContext.from_runtime(
        [
            {"role": "assistant", "content": "What time do you wake up?"},
            {"role": "user", "content": "I wake up at 5:30."},
        ]
        + [
            {"role": "user", "content": ("yoga " * 80) + str(i)}
            for i in range(40)
        ]
    )
    kept, covered = long_ctx.for_llm()
    assert kept
    assert sum(len(item["content"]) for item in kept) <= MAX_CALL_CHARS + 80
    assert any("5:30" in item for item in covered)


def test_ack_turns_are_low_content_but_real_facts_are_not():
    from core.conversation.engagement import is_low_content_turn

    assert is_low_content_turn("Yes. Yes. That's good.")
    assert is_low_content_turn("Yeah. Yeah.")
    assert is_low_content_turn("I I")
    assert not is_low_content_turn("I'm playing cricket.")
    assert not is_low_content_turn("Do you remember my hobbies?")
    assert not is_low_content_turn("Do you know what I am doing in my evening?")
    assert not is_low_content_turn("I am from Bhopal.")
    assert is_low_content_turn("Life.")
    assert not is_low_content_turn("Yoga.")
    assert not is_low_content_turn("Yeah. I don't understand.")


def _daily_topic() -> dict:
    return {
        "_id": "topic_daily",
        "title": "Daily Routine",
        "slug": "a1-daily-routine",
        "level": "A1",
        "goals": [
            {"key": "wake_up", "description": "User can say when they wake up."},
            {"key": "morning", "description": "User can describe a simple morning activity."},
            {"key": "work_or_study_day", "description": "User can say what they do during the day."},
            {"key": "evening", "description": "User can describe an evening activity."},
            {"key": "sleep", "description": "User can say when they go to sleep."},
        ],
    }


def test_morning_chapter_stays_until_both_slots_are_done():
    from core.conversation.session_board import pin_session_goals, update_call_board

    repo = FakeRepo()
    repo.topics = [_daily_topic()]
    cid = _seed(repo, topic_id="topic_daily", progress=0)
    repo.progress[0]["goalsCompleted"] = []
    repo.progress[0]["goalsRemaining"] = [
        "wake_up",
        "morning",
        "work_or_study_day",
        "evening",
        "sleep",
    ]
    analyzer = RecordingAnalyzer({"text": "Nice. What else do you do after you wake up?"})
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I wake up at 5:30.")
    assert state["currentGoalId"] == "morning"
    prompt = analyzer.users[-1]
    assert "Stay on this chapter until it is done: morning" in prompt
    assert "evening" not in prompt.split("User just said:")[0].lower()
    state = svc.handleUserTurn(cid, "Then I do yoga and I take a shower.")
    assert "morning" in (state.get("goalsCompleted") or [])
    assert "wake_up" in (state.get("goalsCompleted") or [])
    assert state["currentGoalId"] == "work_or_study_day"
    state = svc.handleUserTurn(cid, "After that I go to the office.")
    assert state["currentGoalId"] == "evening"
    assert repo.writes["progress"] == 0

    board = update_call_board(
        board=None,
        user_text="After waking I do yoga.",
        topic_goals=_daily_topic()["goals"],
        current_goal_id="wake_up",
    )
    pinned = pin_session_goals(
        topic_goals=_daily_topic()["goals"],
        goals_completed=[],
        goals_remaining=["wake_up", "morning", "work_or_study_day", "evening", "sleep"],
        board=board,
    )
    assert pinned["currentGoalId"] == "wake_up"
    assert pinned["callChapter"] == "morning"
    assert "work_or_study_day" not in pinned["goalsCompleted"]
    assert pinned["goalsRemaining"][0] == "wake_up"


def test_memory_probe_lists_only_this_call_and_skips_mongo():
    repo = FakeRepo()
    repo.topics = [_daily_topic()]
    cid = _seed(repo, topic_id="topic_daily", progress=0)
    repo.progress[0]["goalsCompleted"] = []
    repo.progress[0]["goalsRemaining"] = [
        "wake_up",
        "morning",
        "work_or_study_day",
        "evening",
        "sleep",
    ]
    analyzer = RecordingAnalyzer(
        {"text": "So far you told me you do yoga in the morning. You have not told me your evening yet."}
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "I wake up at 5:30 and then I do yoga.")
    state = svc.handleUserTurn(cid, "Do you know what I am doing in my evening?")
    prompt = analyzer.users[-1]
    assert "So far this call they told you:" in prompt
    assert "yoga" in prompt.lower()
    assert "Never say you don't know" in prompt
    assert "evening" not in (state.get("goalsCompleted") or [])
    assert repo.writes["progress"] == 0
    assert _raw(svc, cid).get("lastUserIntent") == "MEMORY_PROBE"


def test_follow_spoken_chapter_not_wake_up_or_sleep():
    repo = FakeRepo()
    repo.topics = [_daily_topic()]
    cid = _seed(repo, topic_id="topic_daily", progress=0)
    repo.progress[0]["goalsCompleted"] = []
    repo.progress[0]["goalsRemaining"] = [
        "wake_up",
        "morning",
        "work_or_study_day",
        "evening",
        "sleep",
    ]
    analyzer = RecordingAnalyzer({"text": "Okay. What do you do after that?"})
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "taking shower, and after that, am, doing yoga.")
    assert "morning" in (state.get("goalsCompleted") or [])
    assert state["currentGoalId"] == "wake_up"
    assert _raw(svc, cid).get("callBoard", {}).get("chapter") == "morning"
    state = svc.handleUserTurn(cid, "I am in office in the evening time.")
    assert "evening" in (state.get("goalsCompleted") or [])
    assert state["currentGoalId"] != "sleep"
    state = svc.handleUserTurn(cid, "I am. I think I wake up at 06:00 in the morning time.")
    assert _raw(svc, cid).get("callBoard", {}).get("chapter") == "morning"
    state = svc.handleUserTurn(cid, "What happened? I don't get your point.")
    prompt = analyzer.users[-1]
    assert _raw(svc, cid).get("lastUserIntent") == "CONFUSION"
    assert "simpler words" in prompt
    assert repo.writes["progress"] == 0


def test_stt_merge_keeps_both_thoughts_and_drops_garbage():
    from core.conversation.stt_merge import merge_pending_stt

    merged = merge_pending_stt(
        "I'm doing, my press",
        "taking shower, and after that, am, doing yoga.",
    )
    assert "press" in merged.lower()
    assert "yoga" in merged.lower()
    merged = merge_pending_stt(
        "Okay. I am a I I am soft at",
        "developer. I'm doing all the coding.",
    )
    assert "soft" in merged.lower()
    assert "developer" in merged.lower()
    merged = merge_pending_stt(
        "developer. I'm doing all the coding.",
        "Life.",
    )
    assert "developer" in merged.lower()
    assert "life" not in merged.lower()


def test_correction_repeat_is_fuzzy_not_exact():
    from core.conversation.correction import (
        extract_presented_correction,
        repeat_accepted,
        resolve_awaiting_repeat,
        empty_correction_state,
    )

    target = "I wake up at 5."
    assert repeat_accepted("I wake up at five.", target)
    assert repeat_accepted("I usually wake up at five.", target)
    assert repeat_accepted("I wake up at 5 and then I do yoga.", target)
    assert not repeat_accepted("I usually do yoga after that.", target)
    assert extract_presented_correction(
        "Nice! You can say, 'I wake up at 5.' Please repeat it once."
    ) == "I wake up at 5"
    waiting = {
        **empty_correction_state(),
        "status": "awaiting_repeat",
        "correctedText": target,
    }
    assert resolve_awaiting_repeat(
        waiting, user_text="I wake up at five.", user_intent="ANSWER"
    )["lastOutcome"] == "accepted"
    assert resolve_awaiting_repeat(
        waiting, user_text="I usually do yoga after that.", user_intent="ANSWER"
    )["lastOutcome"] == "dismissed"
    assert resolve_awaiting_repeat(
        waiting, user_text="No.", user_intent="ANSWER"
    )["lastOutcome"] == "dismissed"


def test_correction_coach_enters_awaiting_repeat_from_spoken_reply():
    from core.conversation.correction import empty_correction_state

    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer(
        {"text": "Nice! You can say, 'I wake up at 5.' Please repeat it once."}
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    state = svc.handleUserTurn(cid, "I waking up at 5.")
    cs = _raw(svc, cid)["correctionState"]
    assert cs["status"] == "awaiting_repeat"
    assert "wake up at 5" in (cs.get("correctedText") or "").lower()
    assert "Please repeat" in (state["lastAssistantMessage"] or "")
    assert repo.writes["progress"] == 0

    svc.graph.app.update_state(
        {"configurable": {"thread_id": cid}},
        {
            "correctionState": {
                **empty_correction_state(),
                "status": "awaiting_repeat",
                "correctedText": "I wake up at 5.",
                "correctionsGivenThisSession": 1,
                "lastCorrectionAt": 1,
            }
        },
    )
    analyzer.payload = {"text": "Perfect! What do you usually do after that?"}
    svc.handleUserTurn(cid, "I wake up at five.")
    assert _raw(svc, cid)["correctionState"]["status"] == "idle"
    assert "repeated the corrected sentence" in analyzer.users[-1]


def test_correction_coach_drops_repeat_when_user_continues():
    from core.conversation.correction import empty_correction_state

    repo = FakeRepo()
    cid = _seed(repo)
    analyzer = RecordingAnalyzer(
        {"text": "Nice! What kind of yoga do you usually do?"}
    )
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.graph.app.update_state(
        {"configurable": {"thread_id": cid}},
        {
            "correctionState": {
                **empty_correction_state(),
                "status": "awaiting_repeat",
                "correctedText": "I wake up at 5.",
                "correctionsGivenThisSession": 1,
                "lastCorrectionAt": 1,
            }
        },
    )
    svc.handleUserTurn(cid, "I usually do yoga after that.")
    assert _raw(svc, cid)["correctionState"]["status"] == "idle"
    assert "Do not ask them to repeat" in analyzer.users[-1]
    assert repo.writes["progress"] == 0


def test_already_told_you_moves_off_repeated_goal():
    from core.conversation.response import is_similar_question

    assert is_similar_question(
        "What time do you usually wake up?",
        "What time do you wake up in the morning?",
    )
    assert not is_similar_question(
        "What do you usually do in the evening?",
        "What time do you usually wake up?",
    )

    repo = FakeRepo()
    repo.topics = [_daily_topic()]
    cid = _seed(repo, topic_id="topic_daily", progress=0)
    repo.progress[0]["goalsCompleted"] = []
    repo.progress[0]["goalsRemaining"] = [
        "wake_up",
        "morning",
        "work_or_study_day",
        "evening",
        "sleep",
    ]
    analyzer = RecordingAnalyzer({"text": "What time do you usually wake up?"})
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(cid, "I wake up at 5:30.")
    svc.handleUserTurn(cid, "Then I do yoga and I take a shower.")
    state = svc.handleUserTurn(cid, "I already told you I am doing yoga.")
    spoken = (state.get("lastAssistantMessage") or "").lower()
    assert "wake" not in spoken
    assert "already" in spoken or "right" in spoken
    assert state["currentGoalId"] == "work_or_study_day"
    assert repo.writes["progress"] == 0


def test_covering_all_goals_switches_to_next_topic_in_same_call():
    daily = {**_daily_topic(), "order": 1, "isActive": True}
    family = {
        "_id": "topic_family",
        "title": "Family and Friends",
        "slug": "a1-family-friends",
        "level": "A1",
        "order": 2,
        "isActive": True,
        "goals": [
            {"key": "family_members", "description": "User can name family members."},
        ],
    }
    repo = FakeRepo()
    repo.topics = [daily, family]
    cid = _seed(repo, topic_id="topic_daily", progress=0)
    repo.progress[0]["goalsCompleted"] = []
    repo.progress[0]["goalsRemaining"] = [
        "wake_up",
        "morning",
        "work_or_study_day",
        "evening",
        "sleep",
    ]
    analyzer = RecordingAnalyzer({"text": "Nice. What do you do after that?"})
    svc = _svc(repo, analyzer)
    svc.initializeConversationRuntime(cid)
    svc.handleUserTurn(
        cid, "I wake up at 5:30 then I do yoga and I go to the office."
    )
    state = svc.handleUserTurn(
        cid, "In the evening I watch TV and I sleep at 11."
    )
    assert state["topicId"] == "topic_family"
    assert state["topicTitle"] == "Family and Friends"
    practiced = state.get("practicedTopics") or []
    assert practiced
    assert practiced[0]["topicId"] == "topic_daily"
    assert practiced[0]["progress"] == 100
    assert state["currentGoalId"] == "family_members"
    assert repo.writes["progress"] == 0


