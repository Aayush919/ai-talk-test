"""Compact live voice prompt — spoken text only. No JSON schema."""

from __future__ import annotations

from typing import Any

from core.conversation.chat_context import CallChatContext
from core.conversation.correction import correction_prompt_lines
from core.conversation.live_facts import covered_goal_keys
from core.conversation.session_board import CHAPTERS, board_known_lines
from core.semantic.retrieval import compact_memory_context

LIVE_RECENT_MESSAGES = 8  # fallback alias; live path uses CallChatContext
LIVE_MEMORY_ITEMS = 5
LIVE_PROFILE_FACTS = 10
LIVE_ASKED_QUESTIONS = 12

LIVE_SYSTEM_PROMPT = """
You are a friendly English speaking coach in a live voice call.

Have a natural conversation while helping the learner practice the current topic and goal.

Rules:
- This call's conversation is ground truth. If they already answered something, do not ask it again.
- Stay on the current chapter. Follow what they just talked about. Do not drag them back to an old unanswered slot.
- Morning means wake-up and morning routine; finish both before work, evening, or sleep.
- Speak naturally. Usually 1-3 short sentences.
- Ask at most one meaningful follow-up when useful.
- Respond to what they just said. Do not sound like a questionnaire.
- Do not repeat a question listed as already asked, even with different wording.
- If a learner fact is already known, do not ask for it again. Acknowledge it and move to the next remaining goal in this chapter.
- If they ask whether you know or remember something, answer only from this call. Say what they already told you. Never say you don't know if So far this call has it.
- If they say you are repeating, acknowledge and ask a new question in the same chapter.
- Yeah, okay, and short acknowledgements are not a request for a new question.
- Use relevant learner context naturally. Never say you have a memory or a stored profile.
- Do not announce internals.
- Never invent learner facts. Current conversation beats stored profile/memory.
- If they refuse a topic, do not pressure them.
- If they don't know, give a simpler, more specific question.
- If they go slightly off-topic, follow a little, then return.
- If a correction is useful, recast once in the same reply and ask them to repeat it once. Never lecture. Never correct wanna, gonna, or yeah.
- Do not correct every mistake. Conversation first.

Return ONLY the spoken response. No JSON, markdown, labels, analysis, or metadata.
""".strip()

# Back-compat alias — live path uses the compact prompt, not the old JSON contract.
GENERATE_SYSTEM_PROMPT = LIVE_SYSTEM_PROMPT


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _goal_description(state: dict[str, Any]) -> str:
    current = state.get("currentGoalId")
    for item in state.get("topicGoals") or []:
        if isinstance(item, dict) and item.get("key") == current:
            return str(item.get("description") or item.get("key") or "")
    return str((state.get("topicPlan") or {}).get("currentGoalDescription") or "")


def _remaining_goal_labels(state: dict[str, Any]) -> list[dict[str, str]]:
    remaining = [str(item) for item in (state.get("goalsRemaining") or []) if str(item).strip()]
    labels = {
        str(item.get("key")): str(item.get("description") or item.get("key"))
        for item in (state.get("topicGoals") or [])
        if isinstance(item, dict) and item.get("key")
    }
    return [{"key": key, "description": labels.get(key, key.replace("_", " "))} for key in remaining]


def _fact_line(row: Any) -> str:
    if isinstance(row, dict):
        key = _trim(row.get("key") or row.get("label"))
        value = _trim(row.get("value") or row.get("text"))
        if key and value:
            return f"{key}: {value}"
        return value
    return _trim(row)


_ASKED_TOPICS = (
    ("hobbies / free time", ("hobby", "hobbies", "free time", "for fun", "usually do")),
    ("name", ("your name", "call you")),
    ("location", ("where do you live", "where are you from", "which city")),
    ("work or study", ("work", "job", "profession", "study")),
    ("music", ("music", "listen")),
    ("books / reading", ("book", "read")),
)


def _asked_for_prompt(questions: list[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for item in questions:
        text = _trim(item)
        if not text:
            continue
        key = " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(text)
    blob = " ".join(unique).lower()
    topics = [label for label, cues in _ASKED_TOPICS if any(cue in blob for cue in cues)]
    lines = []
    if topics:
        lines.append("topics: " + "; ".join(topics[:8]))
    lines.extend(unique[-LIVE_ASKED_QUESTIONS:])
    return lines


def build_generate_user_prompt(state: dict[str, Any], user_text: str) -> str:
    """Compact live context. Not a LangGraph/Mongo dump."""
    memories = compact_memory_context(
        [
            item
            for item in (state.get("relevantMemories") or [])
            if isinstance(item, dict)
        ]
    )[:LIVE_MEMORY_ITEMS]
    facts = [
        line
        for line in (
            _fact_line(item)
            for item in ((state.get("userContext") or {}).get("profileFacts") or [])[
                :LIVE_PROFILE_FACTS
            ]
        )
        if line
    ]
    utterance = _trim(user_text)
    chat = CallChatContext.from_runtime(
        state.get("recentMessages") or [],
        current_user=utterance,
    )
    recent, covered_qa = chat.for_llm()
    asked = _asked_for_prompt(state.get("recentQuestions") or [])
    entities = [
        str(item) for item in (state.get("lastMentionedEntities") or []) if str(item).strip()
    ][:8]
    remaining = _remaining_goal_labels(state)[:3]
    covered = covered_goal_keys(
        [
            item
            for item in ((state.get("userContext") or {}).get("profileFacts") or [])
            if isinstance(item, dict)
        ],
        state.get("topicGoals") or [],
        state.get("goalsCompleted") or [],
    )
    remaining = [item for item in remaining if item["key"] not in set(covered)]
    chapter = _trim((state.get("callBoard") or {}).get("chapter"))
    if chapter:
        members = next((item for name, item in CHAPTERS if name == chapter), ())
        remaining = [item for item in remaining if item["key"] in set(members)] or remaining
    current_goal = _goal_description(state) or _trim(state.get("currentGoalId")).replace("_", " ")
    current_key = _trim(state.get("currentGoalId"))
    if current_key in set(covered) and remaining:
        current_goal = remaining[0]["description"]
    focus = state.get("currentFocus") or {}
    skill = _trim(focus.get("skill") if isinstance(focus, dict) else "")
    title = _trim(state.get("topicTitle")) or "English conversation"
    level = _trim(state.get("topicLevel"))
    phase = _trim(state.get("conversationPhase")) or "START"
    engagement = _trim(state.get("userEngagement")) or "NORMAL"
    known_this_call = board_known_lines(state.get("callBoard"))
    intent = _trim(state.get("lastUserIntent"))

    lines = [
        f"Current topic: {title}" + (f" ({level})" if level else ""),
    ]
    if chapter:
        lines.append(f"Stay on this chapter until it is done: {chapter}")
    if recent:
        lines.append("This call (do not re-ask anything already answered here):")
        for item in recent:
            role = _trim(item.get("role")) or "user"
            lines.append(f"{role}: {_trim(item.get('content'))}")
    if covered_qa:
        lines.append("Covered this call:")
        lines.extend(f"- {item}" for item in covered_qa)
    if known_this_call:
        lines.append("So far this call they told you:")
        lines.extend(f"- {item}" for item in known_this_call)
    lines.append(f"Current goal: {current_goal}")
    if remaining:
        labels = "; ".join(_trim(item.get("description")) or item["key"] for item in remaining)
        lines.append(f"Remaining goals: {labels}")
    lines.append(f"Conversation phase: {phase}")
    lines.append(f"Learner engagement: {engagement}")
    if skill:
        lines.append(f"Learning focus: {skill.replace('_', ' ')}")
    if facts:
        lines.append("Already known (do not re-ask): " + "; ".join(facts))
    if memories:
        lines.append("Relevant learner context:")
        lines.extend(f"- {item}" for item in memories)
    if asked:
        lines.append("Already asked (do not repeat):")
        lines.extend(f"- {item}" for item in asked)
    if entities:
        lines.append("Recent entities: " + ", ".join(entities))
    if utterance:
        lines.append(f"User just said: {utterance}")
    elif not recent:
        lines.append(
            "The call is starting. Greet warmly and ask one easy question about the current goal. "
            "Do not name it as a lesson."
        )
    if intent == "MEMORY_PROBE":
        lines.append(
            "They asked if you know or remember something. Answer from this call only. "
            "Say 'so far you told me...' using So far this call. "
            "Never say you don't know if that list has it. Do not ask a new question first."
        )
    elif intent == "REPEAT_COMPLAINT":
        lines.append("They said you are repeating. Do not re-ask. Stay on this chapter.")
    elif intent == "CONFUSION":
        cs = state.get("correctionState") or {}
        if _trim(cs.get("status")) != "awaiting_repeat" and _trim(cs.get("lastOutcome")) not in {
            "clarify",
            "accepted",
            "dismissed",
        }:
            lines.append(
                "They did not understand. Repeat the last question in simpler words. "
                "Do not change topic. Do not ask something new."
            )
    lines.extend(
        correction_prompt_lines(
            state.get("correctionState"),
            turn=int(state.get("conversationTurn") or 0),
        )
    )
    lines.append("Speak only the next spoken reply.")
    return "\n".join(lines)
