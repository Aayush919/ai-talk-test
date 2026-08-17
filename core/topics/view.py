"""Compact topic payloads for HTTP / WebSocket. No Mongo writes."""

from __future__ import annotations

from typing import Any


def public_topic(topic: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(topic, dict) or not topic:
        return None
    goals = []
    for item in topic.get("goals") or []:
        if isinstance(item, dict) and item.get("key"):
            goals.append(
                {
                    "key": str(item["key"]),
                    "description": str(item.get("description") or item["key"]),
                }
            )
        elif isinstance(item, str) and item:
            goals.append({"key": item, "description": item.replace("_", " ")})
    return {
        "id": str(topic.get("_id") or topic.get("id") or ""),
        "title": topic.get("title"),
        "slug": topic.get("slug"),
        "level": topic.get("level"),
        "description": topic.get("description"),
        "order": topic.get("order"),
        "goals": goals,
    }


def public_practice_plan(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(plan, dict) or not plan:
        return None
    topic = public_topic(plan.get("topic") if isinstance(plan.get("topic"), dict) else None)
    if topic is None and plan.get("topicId"):
        topic = {
            "id": str(plan.get("topicId")),
            "title": plan.get("topicTitle"),
            "slug": None,
            "level": plan.get("topicLevel"),
            "description": None,
            "order": None,
            "goals": [],
        }
    progress = plan.get("topicProgress") if isinstance(plan.get("topicProgress"), dict) else {}
    return {
        "action": plan.get("action"),
        "topic": topic,
        "currentGoalDescription": plan.get("currentGoalDescription"),
        "progress": int(plan.get("progress") or progress.get("progress") or 0),
        "status": plan.get("topicStatus") or progress.get("status"),
        "initialized": bool(plan.get("initialized")),
    }
