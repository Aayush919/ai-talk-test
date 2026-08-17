"""Structured learning analysis — not used on the live voice path."""

from __future__ import annotations

import json
from typing import Any

LEARNING_ANALYSIS_SYSTEM_PROMPT = """
You are an English learning analysis engine.

Analyze the user's latest completed English conversation.

Identify only meaningful learning signals.

Analyze:
1. Grammar
2. Vocabulary
3. Pronunciation (ONLY if reliable pronunciation data exists)
4. Fluency
5. Comprehension
6. Sentence formation
7. Speaking confidence
8. Strengths
9. Improvement areas
10. Learning patterns

IMPORTANT RULES:
- Do not treat one minor mistake as a recurring weakness.
- Do not invent mistakes.
- Do not infer pronunciation issues from transcript text.
- Do not diagnose the user.
- Do not store temporary emotional states as learning patterns.
- Do not store topic progress (e.g. 3/5 goals completed).
- Do not store personal profile facts (name, profession, hobbies) as learning issues.
- Consider previous learning memory before identifying recurring issues.
- Focus on useful English-learning patterns.
- Do not set frequency or database status. The backend aggregates those.
- Return structured JSON only.

Allowed categories:
grammar, vocabulary, pronunciation, fluency, comprehension, sentenceFormation, confidence

Severity: low | medium | high

Return ONLY this JSON shape:
{
  "signals": [
    {
      "category": "grammar",
      "skill": "past_tense",
      "issue": "difficulty using past tense consistently",
      "severity": "medium",
      "confidence": 0.91
    }
  ],
  "strengths": [
    {
      "skill": "vocabulary",
      "description": "Good everyday vocabulary",
      "confidence": 0.86
    }
  ],
  "patterns": [
    {
      "description": "User gives longer answers when specific follow-up questions are provided.",
      "confidence": 0.83
    }
  ],
  "overallAssessment": {
    "level": "A2",
    "confidence": 0.72,
    "fluency": 0.61,
    "accuracy": 0.58,
    "vocabulary": 0.68,
    "pronunciation": 0.74
  }
}
""".strip()


def build_learning_analysis_prompt(
    *,
    existing_memory: dict[str, Any] | None,
    summary: dict[str, Any],
    pronunciation_evidence: bool,
) -> str:
    compact = _compact_memory(existing_memory)
    summary_text = str(summary.get("summary") or "").strip() or "(empty)"
    mistakes = json.dumps(summary.get("mistakes") or [], ensure_ascii=False, default=str)
    grammar = json.dumps(
        summary.get("grammarPatterns") or [], ensure_ascii=False, default=str
    )
    fluency = json.dumps(
        summary.get("fluencyObservations") or [], ensure_ascii=False, default=str
    )
    strengths = json.dumps(summary.get("strengths") or [], ensure_ascii=False, default=str)
    weaknesses = json.dumps(
        summary.get("weaknesses") or [], ensure_ascii=False, default=str
    )
    evidence = "yes" if pronunciation_evidence else "no"
    return (
        f"CURRENT LEARNING MEMORY:\n{json.dumps(compact, ensure_ascii=False, default=str)}\n\n"
        f"LATEST CONVERSATION SUMMARY:\n{summary_text}\n\n"
        f"SUMMARY MISTAKES:\n{mistakes}\n\n"
        f"GRAMMAR PATTERNS:\n{grammar}\n\n"
        f"FLUENCY OBSERVATIONS:\n{fluency}\n\n"
        f"SUMMARY STRENGTHS:\n{strengths}\n\n"
        f"SUMMARY WEAKNESSES:\n{weaknesses}\n\n"
        f"PRONUNCIATION EVIDENCE AVAILABLE: {evidence}\n"
    )


def _compact_memory(existing: dict[str, Any] | None) -> dict[str, Any]:
    if not existing:
        return {}
    mistakes = []
    for row in existing.get("recurringMistakes") or []:
        if not isinstance(row, dict):
            continue
        mistakes.append(
            {
                "category": row.get("category"),
                "skill": row.get("skill"),
                "issue": row.get("issue"),
                "status": row.get("status"),
                "severity": row.get("severity"),
                "frequency": row.get("frequency"),
                "confidence": row.get("confidence"),
            }
        )
    strengths = []
    for row in existing.get("strengths") or []:
        if not isinstance(row, dict):
            continue
        strengths.append(
            {
                "skill": row.get("skill"),
                "description": row.get("description") or row.get("strength"),
                "frequency": row.get("frequency"),
                "confidence": row.get("confidence"),
            }
        )
    patterns = []
    for row in existing.get("learningPatterns") or []:
        if not isinstance(row, dict):
            continue
        patterns.append(
            {
                "pattern": row.get("pattern") or row.get("description"),
                "frequency": row.get("frequency"),
                "confidence": row.get("confidence"),
            }
        )
    return {
        "recurringMistakes": mistakes,
        "strengths": strengths,
        "learningPatterns": patterns,
        "overallAssessment": existing.get("overallAssessment") or {},
    }
