"""Compact live voice prompt — spoken text only. No JSON schema."""

from __future__ import annotations

from typing import Any

from core.semantic.retrieval import compact_memory_context

LIVE_RECENT_MESSAGES = 8
LIVE_MEMORY_ITEMS = 5
LIVE_PROFILE_FACTS = 3

LIVE_SYSTEM_PROMPT = """
You are a friendly English speaking coach in a live voice call.

Have a natural conversation while helping the learner practice the current topic and goal.

Rules:
- Speak naturally. Usually 1-3 short sentences.
- Ask at most one meaningful follow-up when useful.
- Respond to what they just said. Do not sound like a questionnaire.
- Do not repeat a question listed as already asked.
- Use relevant learner context naturally. Never say you have a memory or a stored profile.
- Do not announce internals.
- Never invent learner facts. Current conversation beats stored profile/memory.
- If they refuse a topic, do not pressure them.
- If they don't know, give a simpler, more specific question.
- If they go slightly off-topic, follow a little, then return.
- If a correction is necessary, keep it brief and natural inside the reply. No grammar lectures.
- Do not correct informal English (wanna, gonna).

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
    recent = [
        item
        for item in (state.get("recentMessages") or [])
        if isinstance(item, dict) and _trim(item.get("content"))
    ][-LIVE_RECENT_MESSAGES:]
    asked = [str(item) for item in (state.get("recentQuestions") or []) if str(item).strip()][-6:]
    entities = [
        str(item) for item in (state.get("lastMentionedEntities") or []) if str(item).strip()
    ][:8]
    remaining = _remaining_goal_labels(state)[:3]
    focus = state.get("currentFocus") or {}
    skill = _trim(focus.get("skill") if isinstance(focus, dict) else "")
    title = _trim(state.get("topicTitle")) or "English conversation"
    level = _trim(state.get("topicLevel"))
    goal = _goal_description(state) or _trim(state.get("currentGoalId")).replace("_", " ")
    phase = _trim(state.get("conversationPhase")) or "START"
    engagement = _trim(state.get("userEngagement")) or "NORMAL"
    utterance = _trim(user_text)

    lines = [
        f"Current topic: {title}" + (f" ({level})" if level else ""),
        f"Current goal: {goal}",
    ]
    if remaining:
        labels = "; ".join(_trim(item.get("description")) or item["key"] for item in remaining)
        lines.append(f"Remaining goals: {labels}")
    lines.append(f"Conversation phase: {phase}")
    lines.append(f"Learner engagement: {engagement}")
    if skill:
        lines.append(f"Learning focus: {skill.replace('_', ' ')}")
    if facts:
        lines.append("Learner: " + "; ".join(facts))
    if memories:
        lines.append("Relevant learner context:")
        lines.extend(f"- {item}" for item in memories)
    if asked:
        lines.append("Already asked (do not repeat):")
        lines.extend(f"- {item}" for item in asked)
        lines.append("recentQuestions")
    if entities:
        lines.append("Recent entities: " + ", ".join(entities))
    if recent:
        lines.append("Recent conversation:")
        for item in recent:
            role = _trim(item.get("role")) or "user"
            lines.append(f"{role}: {_trim(item.get('content'))}")
    if utterance:
        lines.append(f"User just said: {utterance}")
    else:
        lines.append(
            "The call is starting. Greet warmly and ask one easy question about the current goal. "
            "Do not name it as a lesson."
        )
    lines.append("Speak only the next spoken reply.")
    return "\n".join(lines)
