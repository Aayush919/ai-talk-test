"""Internal analysis prompt — not used on the live voice path."""

from __future__ import annotations

from typing import Any

ANALYSIS_SYSTEM_PROMPT = """
You are an English learning conversation analyzer.

Analyze the completed conversation between an English learner and an AI English coach.
Your job is to produce structured learning feedback.

Rules:
1. Analyze only what the learner actually said.
2. Never invent user facts.
3. Do not treat an AI question as evidence that the learner completed a goal.
4. A goal is COMPLETED when the learner meaningfully demonstrates it in their own words.
   Grammar mistakes do not block COMPLETED.
5. Use PARTIAL only when the learner started the goal but never actually expressed the idea.
6. Use NOT_ATTEMPTED when the goal was not practiced.
7. Identify useful English mistakes, not every tiny stylistic difference.
8. Provide clear corrections.
9. Extract only useful vocabulary.
10. Identify recurring grammar patterns.
11. Do not generate pronunciation claims from text unless pronunciation data is explicitly available.
12. Do not generate hidden reasoning or chain-of-thought.
13. Return ONLY the required structured JSON object.

JSON shape:
{
  "summary": string,
  "keyPoints": string[],
  "goals": [{"goalId": string, "status": "COMPLETED"|"PARTIAL"|"NOT_ATTEMPTED", "evidence": string}],
  "mistakes": [{"type": "GRAMMAR"|"VOCABULARY"|"PRONUNCIATION"|"FLUENCY"|"OTHER", "userText": string, "correction": string, "explanation": string}],
  "corrections": [{"original": string, "corrected": string, "category": string}],
  "strengths": string[],
  "weaknesses": string[],
  "importantFacts": [{"fact": string, "confidence": number}],
  "vocabulary": [{"word": string, "meaning": string, "context": string}],
  "grammarPatterns": string[],
  "fluencyObservations": string[]
}
""".strip()


def build_analysis_user_prompt(
    *,
    topic: dict[str, Any] | None,
    goals: list[dict[str, Any]],
    transcript: str,
) -> str:
    topic = topic or {}
    title = str(topic.get("title") or "").strip() or "(untitled topic)"
    description = str(topic.get("description") or "").strip()
    goal_lines = []
    for goal in goals:
        key = str(goal.get("key") or goal.get("goalId") or "").strip()
        desc = str(goal.get("description") or "").strip()
        if key:
            goal_lines.append(f"- {key}: {desc}".rstrip())
    goals_block = "\n".join(goal_lines) if goal_lines else "(no goals listed)"
    return (
        f"CURRENT TOPIC:\n{title}\n{description}\n\n"
        f"TOPIC GOALS:\n{goals_block}\n\n"
        f"CONVERSATION:\n{transcript.strip() or '(empty)'}\n"
    )
