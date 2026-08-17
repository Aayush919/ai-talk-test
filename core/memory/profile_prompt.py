"""Structured profile extraction — not used on the live voice path."""

from __future__ import annotations

import json
from typing import Any

PROFILE_EXTRACTION_SYSTEM_PROMPT = """
You are extracting stable user profile information from an English-learning conversation.

Extract ONLY information that:
1. Was explicitly stated by the user.
2. Is useful for future conversations.
3. Is reasonably stable.
4. Can improve personalization.

Do NOT:
- infer facts
- invent information
- store temporary conversation details
- store grammar mistakes
- store AI statements as user facts
- store unsupported assumptions
- store topic progress (e.g. 2/5 goals completed)
- store English mistakes, corrections, or learning patterns
- store short-lived mood or greetings

Allowed keys (use only these):
name, profession, education, experience, location, interest, hobby, goal,
nativeLanguage, englishLearningGoal, preferredLearningStyle, communicationPreference

Rules:
- Distinguish USER SAID from AI SAID. An AI guess that the user rejects is not a fact.
- Confirmation of an AI question counts only when the user clearly agrees.
- Do not infer profession from tools or skills (e.g. "I work with React" is not profession).
- Do not infer native language merely because the user spoke that language.
- Native language, name, and goals require an explicit user statement.
- For hobbies/interests, return one fact per item. Prefer the stable item name (e.g. "cricket").
- If a newer explicit statement replaces an old value, return action UPSERT for the new value.
- If nothing stable was stated, return {"facts": []}.

For each candidate fact return:
- key
- value
- confidence (0.0 to 1.0)
- action: UPSERT or IGNORE

Confidence:
- 0.90-1.00 explicitly stated or strongly confirmed
- 0.75-0.89 strongly supported
- below 0.75 do not propose (use IGNORE or omit)

Return ONLY structured JSON:
{
  "facts": [
    {
      "key": "profession",
      "value": "software developer",
      "confidence": 0.98,
      "action": "UPSERT"
    }
  ]
}
""".strip()


def build_profile_extraction_prompt(
    *,
    existing_profile: dict[str, Any] | None,
    summary: str,
    important_facts: list[Any],
) -> str:
    profile_block = json.dumps(existing_profile or {}, ensure_ascii=False, default=str)
    facts_block = json.dumps(important_facts or [], ensure_ascii=False, default=str)
    summary_text = (summary or "").strip() or "(empty)"
    return (
        f"CURRENT USER PROFILE:\n{profile_block}\n\n"
        f"CONVERSATION SUMMARY:\n{summary_text}\n\n"
        f"IMPORTANT FACTS:\n{facts_block}\n"
    )
