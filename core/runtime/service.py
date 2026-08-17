"""LangGraph runtime for one ACTIVE call. Mongo remains the source of truth."""

from __future__ import annotations

from typing import Any, Protocol

from core import call_log
from core.conversations.errors import ConversationNotActive, ConversationNotFound
from core.runtime.config import DEFAULT_RUNTIME_CONFIG, RuntimeConfig
from core.runtime.errors import RuntimeStateInvalid
from core.runtime.graph import (
    RuntimeGraph,
    apply_goal_switch,
    clip_recent_messages,
    parse_conversation_decision,
)
from core.runtime.state import public_runtime_state, validate_runtime_state
from core.semantic.retrieval import (
    build_retrieval_context,
    goal_description_from_topic,
)
from core.topics.engine import build_topic_plan

ACTIVE = "ACTIVE"


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _thread_config(conversation_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": str(conversation_id)}}


def _memory_saver():
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


class RuntimeRepo(Protocol):
    def find_conversation_session(self, conversation_id: str) -> dict[str, Any] | None: ...
    def find_topic(self, topic_id: Any) -> dict[str, Any] | None: ...
    def find_progress(self, user_id: str, topic_id: Any) -> dict[str, Any] | None: ...
    def find_user_profile(self, user_id: str) -> dict[str, Any] | None: ...
    def find_learning_memory(self, user_id: str) -> dict[str, Any] | None: ...


class ConversationRuntimeService:
    def __init__(
        self,
        repo: RuntimeRepo,
        *,
        analyzer: Any = None,
        config: RuntimeConfig | dict | None = None,
        checkpointer: Any = None,
        semantic: Any = None,
        tenant_id: str = "talkengly",
    ) -> None:
        self.repo = repo
        if isinstance(config, RuntimeConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = RuntimeConfig.from_mapping(config)
        else:
            self.config = DEFAULT_RUNTIME_CONFIG
        self.checkpointer = checkpointer or _memory_saver()
        self.graph = RuntimeGraph(
            repo,
            analyzer=analyzer,
            config=self.config,
            checkpointer=self.checkpointer,
        )
        self.semantic = semantic
        self.tenant_id = tenant_id or "talkengly"
        self._preview_meta: dict[str, dict[str, Any]] = {}

    def initializeConversationRuntime(self, conversationId: str) -> dict[str, Any]:
        cid = _trim(conversationId)
        if not cid:
            raise ConversationNotFound()
        existing = self._checkpoint_values(cid)
        if existing.get("userId") and existing.get("conversationId"):
            if existing.get("shouldContinue") is False:
                raise ConversationNotActive()
            return public_runtime_state(existing)
        session = self.repo.find_conversation_session(cid)
        if session is None:
            raise ConversationNotFound()
        if session.get("status") != ACTIVE:
            raise ConversationNotActive()
        user_id = _trim(session.get("userId"))
        topic_id = session.get("topicId")
        if not user_id or topic_id is None:
            raise RuntimeStateInvalid()
        topic = self.repo.find_topic(topic_id)
        if topic is None:
            raise RuntimeStateInvalid()
        progress = self.repo.find_progress(user_id, topic_id) or {}
        profile = None
        learning = None
        find_profile = getattr(self.repo, "find_user_profile", None)
        find_learning = getattr(self.repo, "find_learning_memory", None)
        if callable(find_profile):
            profile = find_profile(user_id)
        if callable(find_learning):
            learning = find_learning(user_id)
        payload = {
            "runtimeMode": "init",
            "conversationId": cid,
            "shouldContinue": True,
            "contextSession": {
                "userId": user_id,
                "topicId": str(topic_id),
                "status": session.get("status"),
            },
            "contextTopic": _jsonable(
                {
                    "_id": str(topic.get("_id") if topic.get("_id") is not None else topic_id),
                    "title": topic.get("title"),
                    "level": topic.get("level"),
                    "goals": topic.get("goals") or [],
                }
            ),
            "contextProgress": _jsonable(
                {
                    "status": progress.get("status"),
                    "progress": progress.get("progress") or 0,
                    "goalsCompleted": progress.get("goalsCompleted") or [],
                    "goalsRemaining": progress.get("goalsRemaining") or [],
                }
            ),
            "contextProfile": _jsonable(
                {
                    "profile": (profile or {}).get("profile") or {},
                    "facts": (profile or {}).get("facts") or [],
                }
            )
            if profile
            else None,
            "contextLearning": _jsonable(
                {"recurringMistakes": (learning or {}).get("recurringMistakes") or []}
            )
            if learning
            else None,
            "contextTopicPlan": _jsonable(
                build_topic_plan(
                    topic,
                    progress,
                    action="RESUME"
                    if (progress.get("goalsCompleted") or progress.get("lastConversationId"))
                    else "START",
                )
            ),
            "contextRelevantMemories": [],
            "contextMemoryRetrievalKey": "",
        }
        memories, key = self._retrieve_memories(
            user_id=user_id,
            topic_id=str(topic_id),
            topic_title=_trim(topic.get("title")),
            topic_level=_trim(topic.get("level")),
            current_goal=_first_remaining(progress),
            goal_description=goal_description_from_topic(topic, _first_remaining(progress)),
            learning=learning,
        )
        payload["contextRelevantMemories"] = memories
        payload["contextMemoryRetrievalKey"] = key
        result = self.graph.app.invoke(payload, config=_thread_config(cid))
        return public_runtime_state(result)

    def handleUserTurn(
        self,
        conversationId: str,
        userText: str,
        sttConfidence: float | None = None,
    ) -> dict[str, Any]:
        cid = _trim(conversationId)
        if not cid:
            raise ConversationNotFound()
        current = self._require_active(cid)
        payload = {
            "runtimeMode": "turn",
            "conversationId": cid,
            "incomingUserMessage": _trim(userText),
            "incomingSttConfidence": sttConfidence,
            "shouldContinue": True,
        }
        result = self.graph.app.invoke(payload, config=_thread_config(cid))
        public = public_runtime_state(result)
        refreshed = self._refresh_memories_if_needed(
            cid,
            current,
            current_goal=_trim(public.get("currentGoalId")),
            topic_id=_trim(public.get("topicId")),
            topic_title=_trim(public.get("topicTitle")),
            topic_level=_trim(public.get("topicLevel")),
            user_id=_trim(public.get("userId")),
        )
        return refreshed or public

    def generateOpening(self, conversationId: str) -> dict[str, Any]:
        """First spoken line after runtime init. One LLM call. No Mongo writes."""
        cid = _trim(conversationId)
        state = dict(self._require_active(cid))
        if _trim(state.get("lastAssistantMessage")):
            return {
                "text": _trim(state.get("lastAssistantMessage")),
                "response": _trim(state.get("lastAssistantMessage")),
                "intent": "FOLLOW_UP",
                "question": state.get("lastAssistantQuestion"),
            }
        state["incomingUserMessage"] = ""
        state["lastUserMessage"] = ""
        state["conversationPhase"] = state.get("conversationPhase") or "START"
        generated = self.graph.generate_response(state)
        decision = generated.get("lastDecision") or {}
        text = _trim(
            decision.get("text")
            or decision.get("response")
            or generated.get("lastAssistantMessage")
        )
        update = {
            "lastAssistantMessage": text,
            "lastAssistantQuestion": generated.get("lastAssistantQuestion"),
            "lastDecision": decision,
            "recentMessages": generated.get("recentMessages") or state.get("recentMessages"),
            "conversationPhase": generated.get("conversationPhase") or "WARMUP",
            "recentQuestions": generated.get("recentQuestions"),
            "recentQuestionTypes": generated.get("recentQuestionTypes"),
            "correctionState": generated.get("correctionState"),
            "shouldContinue": True,
            "runtimeMode": "ready",
        }
        self.graph.app.update_state(_thread_config(cid), update)
        return {
            "text": text,
            "response": text,
            "intent": decision.get("intent") or "FOLLOW_UP",
            "question": decision.get("question") or generated.get("lastAssistantQuestion"),
            "llm_ms": int(generated.get("llmMs") or 0),
        }

    def previewResponse(
        self,
        conversationId: str,
        userText: str,
        sttConfidence: float | None = None,
    ) -> dict[str, Any]:
        """Generate a reply without writing the checkpoint (speculative STT)."""
        cid = _trim(conversationId)
        state = dict(self._require_active(cid))
        state["incomingUserMessage"] = _trim(userText)
        if sttConfidence is not None:
            state["incomingSttConfidence"] = sttConfidence
        processed = self.graph.process_user_response(state)
        state.update(processed)
        updated = self.graph.update_runtime_state(state)
        state.update(updated)
        generated = self.graph.generate_response(state)
        decision = generated.get("lastDecision") or {}
        parsed = parse_conversation_decision(decision) or decision
        text = _trim(
            parsed.get("text")
            or parsed.get("response")
            or generated.get("lastAssistantMessage")
        )
        self._preview_meta[cid] = {
            "conversationPhase": generated.get("conversationPhase"),
            "userEngagement": processed.get("userEngagement"),
            "lastUserIntent": processed.get("lastUserIntent"),
            "lastUserAnswer": processed.get("lastUserAnswer"),
            "lastMentionedEntities": processed.get("lastMentionedEntities"),
            "lastAssistantQuestion": generated.get("lastAssistantQuestion"),
            "recentQuestions": generated.get("recentQuestions"),
            "recentQuestionTypes": generated.get("recentQuestionTypes"),
            "correctionState": generated.get("correctionState"),
            "pendingGoalEvidence": generated.get("pendingGoalEvidence"),
            "lastDecision": decision,
            "shouldContinue": generated.get("shouldContinue", True),
            "sttConfidence": processed.get("sttConfidence"),
            "coachingStrategy": processed.get("coachingStrategy"),
        }
        return {
            "response": text,
            "text": text,
            "intent": parsed.get("intent") or "FOLLOW_UP",
            "targetGoalId": parsed.get("targetGoalId"),
            "followUpNeeded": parsed.get("followUpNeeded", True),
            "question": parsed.get("question"),
            "shouldContinue": bool(generated.get("shouldContinue", True)),
            "shouldTransition": bool(parsed.get("shouldTransition")),
            "goalEvidence": parsed.get("goalEvidence") or generated.get("pendingGoalEvidence"),
            "correction": parsed.get("correction"),
            "conversationPhase": generated.get("conversationPhase"),
            "llm_ms": int(generated.get("llmMs") or 0),
        }

    def applyCommittedTurn(
        self,
        conversationId: str,
        *,
        userText: str,
        assistantText: str,
        targetGoalId: str | None = None,
    ) -> dict[str, Any]:
        """Persist the spoken turn into the checkpoint. No Mongo memory writes."""
        cid = _trim(conversationId)
        current = dict(self._require_active(cid))
        messages = [
            dict(item)
            for item in (current.get("recentMessages") or [])
            if isinstance(item, dict)
        ]
        user_text = _trim(userText)
        assistant_text = _trim(assistantText)
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        messages = clip_recent_messages(messages, self.config.max_recent_messages)
        turn = int(current.get("conversationTurn") or 0)
        if user_text:
            turn += 1
        update = {
            "runtimeMode": "commit",
            "recentMessages": messages,
            "lastUserMessage": user_text or current.get("lastUserMessage"),
            "lastAssistantMessage": assistant_text or current.get("lastAssistantMessage"),
            "conversationTurn": turn,
            "shouldContinue": True,
        }
        switched = apply_goal_switch(current, targetGoalId)
        update.update(switched)
        meta = self._preview_meta.pop(cid, {})
        for key, value in meta.items():
            if key == "lastDecision":
                update[key] = value
                continue
            if value is not None:
                update[key] = value
        self.graph.app.update_state(_thread_config(cid), update)
        latest = dict(self._checkpoint_values(cid))
        refreshed = self._refresh_memories_if_needed(
            cid,
            current,
            current_goal=_trim(latest.get("currentGoalId") or current.get("currentGoalId")),
            topic_id=_trim(latest.get("topicId") or current.get("topicId")),
            topic_title=_trim(latest.get("topicTitle") or current.get("topicTitle")),
            topic_level=_trim(latest.get("topicLevel") or current.get("topicLevel")),
            user_id=_trim(latest.get("userId") or current.get("userId")),
        )
        return refreshed or self.getRuntimeState(cid)

    def endConversationRuntime(self, conversationId: str) -> dict[str, Any]:
        cid = _trim(conversationId)
        if not cid:
            raise ConversationNotFound()
        if not self._checkpoint_values(cid):
            raise ConversationNotFound()
        result = self.graph.app.invoke(
            {"runtimeMode": "end", "shouldContinue": False, "conversationId": cid},
            config=_thread_config(cid),
        )
        return public_runtime_state(result)

    def getRuntimeState(self, conversationId: str) -> dict[str, Any]:
        cid = _trim(conversationId)
        values = self._checkpoint_values(cid)
        if not values.get("conversationId"):
            raise ConversationNotFound()
        return public_runtime_state(values)

    def _require_active(self, conversation_id: str) -> dict[str, Any]:
        values = self._checkpoint_values(conversation_id)
        if not values.get("userId"):
            raise ConversationNotFound()
        if values.get("shouldContinue") is False:
            raise ConversationNotActive()
        validate_runtime_state(values)
        return values

    def _checkpoint_values(self, conversation_id: str) -> dict[str, Any]:
        snap = self.graph.app.get_state(_thread_config(conversation_id))
        values = getattr(snap, "values", None)
        if isinstance(values, dict):
            return values
        return {}

    def _refresh_memories_if_needed(
        self,
        conversation_id: str,
        previous: dict[str, Any],
        *,
        current_goal: str,
        topic_id: str,
        topic_title: str,
        topic_level: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        if self.semantic is None:
            return None
        cfg = getattr(self.semantic, "config", None)
        every_turn = bool(getattr(cfg, "retrieve_every_turn", False))
        on_goal = bool(getattr(cfg, "retrieve_on_goal_change", True))
        on_topic = bool(getattr(cfg, "retrieve_on_topic_change", True))
        if not every_turn and not on_goal and not on_topic:
            return None
        focus = ""
        for row in ((previous.get("userContext") or {}).get("learningSignals") or []):
            if str(row.get("status") or "").upper() == "ACTIVE":
                focus = str(row.get("skill") or "")
                break
        context = build_retrieval_context(
            tenant_id=self.tenant_id,
            user_id=user_id,
            topic_id=topic_id,
            topic_title=topic_title,
            topic_level=topic_level,
            current_goal=current_goal,
            learning_focus=focus,
        )
        previous_key = _trim(previous.get("memoryRetrievalKey"))
        goal_changed = current_goal and current_goal != _trim(previous.get("currentGoalId"))
        topic_changed = topic_id and topic_id != _trim(previous.get("topicId"))
        should = every_turn or (
            context.fingerprint != previous_key
            and ((on_goal and goal_changed) or (on_topic and topic_changed))
        )
        if not should:
            return None
        memories, key = self._retrieve_memories(
            user_id=user_id,
            topic_id=topic_id,
            topic_title=topic_title,
            topic_level=topic_level,
            current_goal=current_goal,
            goal_description="",
            learning=None,
            learning_focus=focus,
        )
        self.graph.app.update_state(
            _thread_config(conversation_id),
            {"relevantMemories": memories, "memoryRetrievalKey": key},
        )
        return self.getRuntimeState(conversation_id)

    def _retrieve_memories(
        self,
        *,
        user_id: str,
        topic_id: str,
        topic_title: str,
        topic_level: str,
        current_goal: str,
        learning: dict[str, Any] | None,
        goal_description: str = "",
        learning_focus: str = "",
    ) -> tuple[list[dict[str, Any]], str]:
        if self.semantic is None:
            return [], ""
        try:
            retrieve_for_runtime = getattr(self.semantic, "retrieveForRuntime", None)
            if callable(retrieve_for_runtime):
                return retrieve_for_runtime(
                    user_id=user_id,
                    topic_id=topic_id,
                    topic_title=topic_title,
                    topic_level=topic_level,
                    current_goal=current_goal,
                    goal_description=goal_description,
                    learning=learning,
                    tenant_id=self.tenant_id,
                )
            context = build_retrieval_context(
                tenant_id=self.tenant_id,
                user_id=user_id,
                topic_id=topic_id,
                topic_title=topic_title,
                topic_level=topic_level,
                current_goal=current_goal,
                goal_description=goal_description,
                learning=learning,
                learning_focus=learning_focus,
            )
            memories = self.semantic.retrieveRelevantMemories(
                tenantId=self.tenant_id,
                userId=user_id,
                query=context.query,
                topicId=topic_id or None,
                topicLevel=topic_level or None,
            )
            return memories, context.fingerprint
        except Exception as exc:  # noqa: BLE001
            call_log.error(
                "MEMORY",
                "QDRANT_RETRIEVAL_FAILED",
                extra={"userId": user_id, "detail": str(exc)[:240]},
            )
            return [], ""


def _first_remaining(progress: dict[str, Any] | None) -> str:
    remaining = (progress or {}).get("goalsRemaining") or []
    if remaining:
        return str(remaining[0])
    return ""
