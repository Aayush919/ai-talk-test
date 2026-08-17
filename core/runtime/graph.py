"""LangGraph nodes for one active call. No permanent Mongo writes."""

from __future__ import annotations

import time
from typing import Any, Protocol

from langgraph.graph import END, START, StateGraph

from core import call_log
from core.conversation.config import DEFAULT_CONVERSATION_CONFIG
from core.conversation.correction import CorrectionService, empty_correction_state
from core.conversation.engagement import detect_engagement, detect_user_intent
from core.conversation.entities import extract_entities
from core.conversation.phase import next_phase
from core.conversation.prompts import GENERATE_SYSTEM_PROMPT, build_generate_user_prompt
from core.conversation.response import (
    coerce_spoken_text,
    decision_from_spoken,
    extract_question,
    parse_ai_response,
    question_key,
    spoken_text_only,
)
from core.runtime.config import RuntimeConfig
from core.runtime.state import (
    RuntimeStateDict,
    validate_runtime_state,
)
from core.text_clean import clip_spoken_reply
from core.topics.engine import select_current_goal

FALLBACK_RESPONSE = "Nice. Tell me a bit more."
PROFILE_FACT_KEYS = (
    "name",
    "profession",
    "education",
    "location",
    "nativeLanguage",
    "englishLearningGoal",
)
FOCUS_FROM_CATEGORY = {
    "grammar": "grammar",
    "vocabulary": "vocabulary",
    "pronunciation": "pronunciation",
    "fluency": "fluency",
}


class RuntimeAnalyzer(Protocol):
    def speak(self, *, system: str, user: str) -> str: ...


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _goal_keys(topic: dict[str, Any] | None) -> list[str]:
    keys: list[str] = []
    for goal in (topic or {}).get("goals") or []:
        if isinstance(goal, dict) and goal.get("key"):
            keys.append(str(goal["key"]))
        elif isinstance(goal, str) and goal:
            keys.append(goal)
    return keys


def _goal_records(topic: dict[str, Any] | None) -> list[dict[str, str]]:
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
            out.append({"key": goal, "description": goal})
    return out


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def clip_recent_messages(
    messages: list[dict[str, str]], limit: int
) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    if len(messages) <= limit:
        return list(messages)
    return list(messages[-limit:])


def select_profile_facts(profile_doc: dict[str, Any] | None, limit: int) -> list[dict[str, str]]:
    if not profile_doc:
        return []
    facts: list[dict[str, str]] = []
    profile = profile_doc.get("profile") if isinstance(profile_doc.get("profile"), dict) else {}
    for key in PROFILE_FACT_KEYS:
        value = _trim(profile.get(key))
        if value:
            facts.append({"key": key, "value": value})
    for key in ("hobbies", "interests"):
        for item in profile.get(key) or []:
            text = _trim(item)
            if text:
                facts.append({"key": key.rstrip("s"), "value": text})
    for row in profile_doc.get("facts") or []:
        if not isinstance(row, dict):
            continue
        key = _trim(row.get("key"))
        value = _trim(row.get("value"))
        if key and value and not any(item["key"] == key and item["value"] == value for item in facts):
            facts.append({"key": key, "value": value})
    return facts[: max(0, limit)]


def select_learning_signals(
    learning_doc: dict[str, Any] | None, limit: int
) -> list[dict[str, str]]:
    if not learning_doc:
        return []
    signals: list[dict[str, str]] = []
    for row in learning_doc.get("recurringMistakes") or []:
        if not isinstance(row, dict):
            continue
        status = _trim(row.get("status")).upper()
        if status not in {"ACTIVE", "IMPROVING"}:
            continue
        skill = _trim(row.get("skill"))
        if not skill:
            continue
        signals.append(
            {
                "skill": skill,
                "status": status,
                "category": _trim(row.get("category")) or "grammar",
            }
        )
    return signals[: max(0, limit)]


def parse_conversation_decision(raw: Any) -> dict[str, Any] | None:
    return parse_ai_response(raw)


def fallback_decision() -> dict[str, Any]:
    return {
        "text": FALLBACK_RESPONSE,
        "response": FALLBACK_RESPONSE,
        "intent": "FOLLOW_UP",
        "targetGoalId": None,
        "followUpNeeded": True,
        "correction": None,
        "shouldContinue": True,
        "shouldTransition": False,
        "question": "Could you tell me a little more?",
        "goalEvidence": None,
        "conversationPhase": None,
    }


def apply_goal_switch(state: dict[str, Any], target_goal_id: str | None) -> dict[str, Any]:
    target = _trim(target_goal_id)
    if not target:
        return {}
    allowed = [
        str(item.get("key"))
        for item in (state.get("topicGoals") or [])
        if isinstance(item, dict) and item.get("key")
    ]
    if not allowed:
        allowed = list(state.get("goalsRemaining") or []) + list(state.get("goalsCompleted") or [])
    if target not in allowed:
        return {}
    index = allowed.index(target) if target in allowed else None
    return {"currentGoalId": target, "currentGoalIndex": index}


class RuntimeGraph:
    def __init__(
        self,
        repo: Any,
        *,
        analyzer: RuntimeAnalyzer | None = None,
        config: RuntimeConfig,
        checkpointer: Any,
    ) -> None:
        self.repo = repo
        self.analyzer = analyzer
        self.config = config
        self.conversation_config = DEFAULT_CONVERSATION_CONFIG
        self.corrections = CorrectionService(self.conversation_config)
        builder = StateGraph(RuntimeStateDict)
        builder.add_node("loadRuntimeContext", self.load_runtime_context)
        builder.add_node("prepareConversation", self.prepare_conversation)
        builder.add_node("processUserResponse", self.process_user_response)
        builder.add_node("updateRuntimeState", self.update_runtime_state)
        builder.add_node("generateResponse", self.generate_response)
        builder.add_node("endRuntime", self.end_runtime)
        builder.add_conditional_edges(
            START,
            self.route_start,
            {
                "loadRuntimeContext": "loadRuntimeContext",
                "processUserResponse": "processUserResponse",
                "endRuntime": "endRuntime",
            },
        )
        builder.add_edge("loadRuntimeContext", "prepareConversation")
        builder.add_edge("prepareConversation", END)
        builder.add_edge("processUserResponse", "updateRuntimeState")
        builder.add_edge("updateRuntimeState", "generateResponse")
        builder.add_edge("generateResponse", END)
        builder.add_edge("endRuntime", END)
        self.app = builder.compile(checkpointer=checkpointer)

    def route_start(self, state: RuntimeStateDict) -> str:
        mode = _trim(state.get("runtimeMode")) or "init"
        if mode == "end" or state.get("shouldContinue") is False and not state.get("userId"):
            return "endRuntime"
        if mode == "end":
            return "endRuntime"
        if mode == "turn":
            return "processUserResponse"
        return "loadRuntimeContext"

    def load_runtime_context(self, state: RuntimeStateDict) -> dict[str, Any]:
        session = state.get("contextSession") or {}
        topic = state.get("contextTopic") or {}
        progress = state.get("contextProgress") or {}
        profile = state.get("contextProfile")
        learning = state.get("contextLearning")
        topic_id = str(session.get("topicId") or topic.get("_id") or "")
        goals = _goal_keys(topic)
        completed = [
            key
            for key in _as_str_list(progress.get("goalsCompleted"))
            if key in goals
        ] if goals else _as_str_list(progress.get("goalsCompleted"))
        remaining = [
            key
            for key in (goals if goals else _as_str_list(progress.get("goalsRemaining")))
            if key not in set(completed)
        ]
        loaded = {
            "userId": str(session.get("userId") or ""),
            "conversationId": str(state.get("conversationId") or ""),
            "topicId": topic_id,
            "topicTitle": _trim(topic.get("title")),
            "topicLevel": _trim(topic.get("level")) or None,
            "topicStatus": _trim(progress.get("status")) or None,
            "topicProgress": int(progress.get("progress") or 0),
            "goalsCompleted": completed,
            "goalsRemaining": remaining,
            "topicGoals": _goal_records(topic),
            "recentMessages": [],
            "lastUserMessage": None,
            "lastAssistantMessage": None,
            "conversationTurn": 0,
            "userContext": {
                "profileFacts": select_profile_facts(profile, self.config.max_profile_facts),
                "learningSignals": select_learning_signals(
                    learning, self.config.max_learning_signals
                ),
            },
            "pendingMemorySignals": [],
            "relevantMemories": list(state.get("contextRelevantMemories") or [])[
                : self.config.max_relevant_memories
            ],
            "memoryRetrievalKey": str(state.get("contextMemoryRetrievalKey") or ""),
            "topicPlan": {
                "action": (state.get("contextTopicPlan") or {}).get("action"),
                "shouldContinueTopic": (state.get("contextTopicPlan") or {}).get(
                    "shouldContinueTopic"
                ),
                "currentGoalDescription": (state.get("contextTopicPlan") or {}).get(
                    "currentGoalDescription"
                ),
            },
            "shouldContinue": True,
            "turnOutcome": None,
            "lastDecision": None,
        }
        validate_runtime_state(loaded)
        return loaded

    def prepare_conversation(self, state: RuntimeStateDict) -> dict[str, Any]:
        remaining = list(state.get("goalsRemaining") or [])
        completed = list(state.get("goalsCompleted") or [])
        topic = {
            "goals": state.get("topicGoals") or [],
            "title": state.get("topicTitle"),
            "level": state.get("topicLevel"),
        }
        current, index, _description = select_current_goal(
            topic,
            {"goalsCompleted": completed, "goalsRemaining": remaining},
        )
        signals = ((state.get("userContext") or {}).get("learningSignals") or [])
        focus = {"type": "conversation"}
        target_skill = None
        if signals:
            first = signals[0]
            category = _trim(first.get("category"))
            focus = {
                "type": FOCUS_FROM_CATEGORY.get(category, "conversation"),
                "skill": first.get("skill"),
            }
            target_skill = first.get("skill")
        return {
            "currentGoalId": current,
            "currentGoalIndex": index,
            "goalsCompleted": completed,
            "goalsRemaining": remaining,
            "currentFocus": focus,
            "coachingStrategy": {
                "correctionMode": "SUBTLE",
                "followUpStyle": "OPEN",
                "targetSkill": target_skill,
            },
            "conversationPhase": "START",
            "userEngagement": "NORMAL",
            "correctionState": empty_correction_state(),
            "recentQuestions": [],
            "recentQuestionTypes": [],
            "lastMentionedEntities": [],
            "pendingGoalEvidence": None,
            "lastAssistantQuestion": None,
            "lastUserAnswer": None,
            "lastUserIntent": None,
            "shouldContinue": True,
            "runtimeMode": "ready",
            "contextSession": None,
            "contextTopic": None,
            "contextProgress": None,
            "contextProfile": None,
            "contextLearning": None,
            "contextTopicPlan": None,
        }

    def process_user_response(self, state: RuntimeStateDict) -> dict[str, Any]:
        text = _trim(state.get("incomingUserMessage"))
        words = [part for part in text.split() if part]
        engagement = detect_engagement(text)
        user_intent = detect_user_intent(text)
        if not text:
            progress = "NOT_STARTED"
            needs_follow = True
        elif len(words) >= 6:
            progress = "GOOD"
            needs_follow = len(words) < 10
        else:
            progress = "IN_PROGRESS"
            needs_follow = True
        strategy = dict(state.get("coachingStrategy") or {})
        if engagement == "LOW":
            strategy["followUpStyle"] = "SPECIFIC"
        elif engagement == "HIGH":
            strategy["followUpStyle"] = "OPEN"
        if user_intent == "CORRECTION_REQUEST":
            strategy["correctionMode"] = "IMMEDIATE"
        else:
            strategy["correctionMode"] = strategy.get("correctionMode") or "SUBTLE"
        stt = state.get("incomingSttConfidence")
        if stt is None:
            stt = state.get("sttConfidence")
        try:
            stt_value = float(stt) if stt is not None else None
        except (TypeError, ValueError):
            stt_value = None
        entities = extract_entities(
            text,
            previous=list(state.get("lastMentionedEntities") or []),
            limit=self.conversation_config.max_entities,
        )
        correction_state = dict(state.get("correctionState") or empty_correction_state())
        correction_state["correctionsGivenThisTurn"] = 0
        return {
            "lastUserMessage": text or None,
            "lastUserAnswer": text or None,
            "lastUserIntent": user_intent,
            "userEngagement": engagement,
            "lastMentionedEntities": entities,
            "sttConfidence": stt_value,
            "incomingSttConfidence": None,
            "coachingStrategy": strategy,
            "correctionState": correction_state,
            "turnOutcome": {
                "goalProgress": progress,
                "needsFollowUp": needs_follow,
                "detectedLearningSignal": False,
            },
            "pendingMemorySignals": [],
        }

    def update_runtime_state(self, state: RuntimeStateDict) -> dict[str, Any]:
        messages = [
            dict(item)
            for item in (state.get("recentMessages") or [])
            if isinstance(item, dict) and _trim(item.get("content"))
        ]
        user_text = _trim(state.get("incomingUserMessage") or state.get("lastUserMessage"))
        if user_text:
            messages.append({"role": "user", "content": user_text})
        turn = int(state.get("conversationTurn") or 0)
        if user_text:
            turn += 1
        messages = clip_recent_messages(messages, self.config.max_recent_messages)
        return {
            "recentMessages": messages,
            "conversationTurn": turn,
            "lastUserMessage": user_text or state.get("lastUserMessage"),
            "incomingUserMessage": None,
        }

    def generate_response(self, state: RuntimeStateDict) -> dict[str, Any]:
        user_text = _trim(state.get("lastUserMessage"))
        decision, llm_ms = self._decide(state, user_text)
        user_intent = _trim(state.get("lastUserIntent")) or "ANSWER"
        stt = state.get("sttConfidence")
        try:
            stt_value = float(stt) if stt is not None else None
        except (TypeError, ValueError):
            stt_value = None
        correction = self.corrections.filter_live_correction(
            decision.get("correction"),
            correction_state=state.get("correctionState"),
            user_intent=user_intent,
            stt_confidence=stt_value,
            pronunciation_evidence=bool(state.get("pronunciationEvidence")),
        )
        decision["correction"] = correction
        reply = clip_spoken_reply(
            spoken_text_only(decision) or FALLBACK_RESPONSE,
            user_text=user_text,
        ) or FALLBACK_RESPONSE
        decision["text"] = reply
        decision["response"] = reply
        question = _trim(decision.get("question")) or extract_question(reply)
        if question:
            decision["question"] = question
        recent_questions = [
            str(item)
            for item in (state.get("recentQuestions") or [])
            if str(item).strip()
        ]
        if question and question_key(question) not in {question_key(item) for item in recent_questions}:
            recent_questions.append(question)
        recent_questions = recent_questions[-self.conversation_config.max_recent_questions :]
        question_types = [
            str(item)
            for item in (state.get("recentQuestionTypes") or [])
            if str(item).strip()
        ]
        qtype = _trim(decision.get("questionType")).upper()
        if qtype:
            question_types.append(qtype)
        question_types = question_types[-8:]
        should_close = user_intent == "GOODBYE" or decision.get("intent") == "CLOSING"
        if user_intent == "GOODBYE":
            decision["shouldContinue"] = False
            decision["intent"] = "CLOSING"
        phase = next_phase(
            state.get("conversationPhase"),
            turn=int(state.get("conversationTurn") or 0),
            engagement=_trim(state.get("userEngagement")) or "NORMAL",
            user_intent=user_intent,
            llm_phase=None,
            should_close=should_close,
            should_transition=False,
        )
        decision["conversationPhase"] = phase
        evidence = {
            "goalId": _trim(state.get("currentGoalId")),
            "coveredAreas": [
                str(item)
                for item in (state.get("lastMentionedEntities") or [])
                if str(item).strip()
            ][:6],
            "remainingAreas": [],
            "completionConfidence": 0.35 if len(user_text.split()) >= 8 else 0.1,
        }
        if not user_text:
            evidence = None
        messages = [
            dict(item)
            for item in (state.get("recentMessages") or [])
            if isinstance(item, dict)
        ]
        messages.append({"role": "assistant", "content": reply})
        messages = clip_recent_messages(messages, self.config.max_recent_messages)
        outcome = dict(state.get("turnOutcome") or {})
        outcome["needsFollowUp"] = bool(decision.get("followUpNeeded", True))
        correction_state = self.corrections.apply_to_state(dict(state), correction)
        return {
            "lastAssistantMessage": reply,
            "lastAssistantQuestion": question or extract_question(reply),
            "lastDecision": decision,
            "recentMessages": messages,
            "turnOutcome": outcome,
            "incomingAssistantMessage": None,
            "conversationPhase": phase,
            "recentQuestions": recent_questions,
            "recentQuestionTypes": question_types,
            "correctionState": correction_state,
            "pendingGoalEvidence": evidence,
            "shouldContinue": bool(decision.get("shouldContinue", True)),
            "llmMs": llm_ms,
        }

    def end_runtime(self, state: RuntimeStateDict) -> dict[str, Any]:
        return {"shouldContinue": False, "runtimeMode": "ended"}

    def _decide(self, state: dict[str, Any], user_text: str) -> tuple[dict[str, Any], int]:
        if self.analyzer is None:
            return fallback_decision(), 0
        attempts = 1 + max(0, self.config.llm_retries)
        last_error: Exception | None = None
        system = GENERATE_SYSTEM_PROMPT
        user = build_generate_user_prompt(state, user_text)
        for _ in range(attempts):
            t0 = time.perf_counter()
            try:
                spoken = self._speak(system=system, user=user)
                llm_ms = int((time.perf_counter() - t0) * 1000)
                text = coerce_spoken_text(spoken)
                if not text:
                    raise RuntimeError("empty live reply")
                return decision_from_spoken(
                    text, user_intent=_trim(state.get("lastUserIntent")) or "ANSWER"
                ), llm_ms
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                llm_ms = int((time.perf_counter() - t0) * 1000)
                call_log.warn(
                    "RUNTIME",
                    f"live speak failed: {type(exc).__name__}",
                    extra={"err": str(exc)[:240], "latency_ms": llm_ms},
                )
        if last_error is not None:
            call_log.warn("RUNTIME", f"decision fallback: {last_error}")
        return fallback_decision(), 0

    def _speak(self, *, system: str, user: str) -> str:
        speak = getattr(self.analyzer, "speak", None)
        if callable(speak):
            return str(speak(system=system, user=user) or "")
        raw = self.analyzer.analyze_json(system=system, user=user)
        parsed = parse_conversation_decision(raw)
        if parsed is not None:
            return spoken_text_only(parsed)
        if isinstance(raw, dict):
            return coerce_spoken_text(raw)
        return coerce_spoken_text(raw)
