"""Structured semantic memory extraction — not used on the live voice path."""

from __future__ import annotations

import json
from typing import Any

SEMANTIC_EXTRACTION_SYSTEM_PROMPT = """
You are a semantic memory extraction engine for an AI English Speaking Coach.

Identify only information that is likely to be useful in future conversations.

Possible memory types:
PROFILE_FACT, LEARNING_PATTERN, LEARNING_WEAKNESS, LEARNING_STRENGTH,
EXPERIENCE, PREFERENCE, CONVERSATION_MEMORY

Rules:
1. Do not store trivial conversation.
2. Do not store every sentence.
3. Do not duplicate existing memories.
4. Do not invent facts.
5. Do not infer pronunciation issues without reliable pronunciation evidence.
6. Do not store temporary runtime state.
7. Prefer stable, reusable information.
8. Learning weaknesses require evidence.
9. Return structured JSON only.
10. Keep each memory concise.
11. Do not store topic progress, transcripts, or greetings.

Return ONLY:
{
  "memories": [
    {
      "memoryType": "LEARNING_WEAKNESS",
      "category": "grammar",
      "skill": "past_tense",
      "content": "User frequently struggles with past tense when describing previous experiences.",
      "importance": 0.82,
      "confidence": 0.91,
      "scope": "GLOBAL"
    }
  ]
}
""".strip()


def build_semantic_extraction_prompt(
    *,
    summary: str,
    existing_memories: list[dict[str, Any]],
    learning: dict[str, Any] | None = None,
) -> str:
    compact = []
    for row in existing_memories[:12]:
        compact.append(
            {
                "memoryType": row.get("memoryType"),
                "skill": row.get("skill"),
                "content": row.get("content"),
            }
        )
    return (
        "EXISTING RELEVANT MEMORIES:\n"
        f"{json.dumps(compact, ensure_ascii=False, default=str)}\n\n"
        "CONVERSATION SUMMARY:\n"
        f"{(summary or '').strip() or '(empty)'}\n\n"
        "LEARNING MEMORY SNAPSHOT:\n"
        f"{json.dumps(learning or {}, ensure_ascii=False, default=str)}\n"
    )
