"""Learning memory — LLM proposes signals; backend aggregates patterns."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Protocol

from core import call_log
from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationNotCompleted,
    ConversationNotFound,
    SummaryNotFound,
)
from core.conversations.summary_service import parse_json_object
from core.memory.errors import LearningMemoryUpdateFailed
from core.memory.learning_config import (
    DEFAULT_LEARNING_MEMORY_CONFIG,
    LearningMemoryConfig,
)
from core.memory.learning_prompt import (
    LEARNING_ANALYSIS_SYSTEM_PROMPT,
    build_learning_analysis_prompt,
)

COMPLETED = "COMPLETED"
STATUS_ACTIVE = "ACTIVE"
STATUS_IMPROVING = "IMPROVING"
STATUS_RESOLVED = "RESOLVED"
SKILL_CATEGORIES = (
    "grammar",
    "vocabulary",
    "pronunciation",
    "fluency",
    "comprehension",
    "sentenceFormation",
    "confidence",
)
ALLOWED_CATEGORIES = frozenset(SKILL_CATEGORIES)
SEVERITIES = frozenset({"low", "medium", "high"})
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1")
PROFILE_SKILLS = frozenset(
    {
        "name",
        "profession",
        "education",
        "experience",
        "hobby",
        "hobbies",
        "interest",
        "interests",
        "goal",
        "native_language",
        "nativelanguage",
        "englishlearninggoal",
    }
)
_TOPIC_POLLUTION = re.compile(
    r"\d+\s*/\s*\d+|goals?\s+completed|topic\s+progress",
    re.I,
)
_CATEGORY_ALIASES = {
    "sentence_formation": "sentenceFormation",
    "sentence-formation": "sentenceFormation",
    "speaking_confidence": "confidence",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm(value: Any) -> str:
    return " ".join(_trim(value).lower().split())


def _skill_slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", _norm(value)).strip("_")
    return text


def _category(value: Any) -> str:
    raw = _norm(value).replace(" ", "_")
    if raw in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[raw]
    if raw == "sentenceformation":
        return "sentenceFormation"
    return raw


def _severity(value: Any) -> str:
    text = _norm(value)
    return text if text in SEVERITIES else "medium"


def _clamp(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return 0.0
    if number > 1:
        return 1.0
    return number


def _processed_ids(doc: dict[str, Any] | None) -> set[str]:
    if not doc:
        return set()
    ids = {str(item) for item in (doc.get("processedConversationIds") or [])}
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    last = _trim(meta.get("lastAnalyzedConversationId") or meta.get("lastProcessedConversationId"))
    if last:
        ids.add(last)
    return ids


def _has_pronunciation_evidence(summary: dict[str, Any]) -> bool:
    if summary.get("pronunciationEvidence") or summary.get("pronunciationData"):
        return True
    for row in summary.get("mistakes") or []:
        if isinstance(row, dict) and _trim(row.get("type")).upper() == "PRONUNCIATION":
            return True
    return False


def _is_pollution(skill: str, issue: str) -> bool:
    blob = f"{skill} {issue}"
    if _TOPIC_POLLUTION.search(blob):
        return True
    if _skill_slug(skill) in PROFILE_SKILLS:
        return True
    return False


def empty_skills() -> dict[str, list]:
    return {key: [] for key in SKILL_CATEGORIES}


def public_learning_memory(
    doc: dict[str, Any] | None, *, user_id: str = ""
) -> dict[str, Any]:
    if not doc:
        return {
            "userId": user_id,
            "skills": empty_skills(),
            "recurringMistakes": [],
            "strengths": [],
            "improvementAreas": [],
            "learningPatterns": [],
            "overallAssessment": {},
            "metadata": {
                "totalAnalyzedConversations": 0,
                "lastAnalyzedConversationId": None,
                "lastAnalyzedAt": None,
            },
            "version": 0,
        }
    skills = dict(empty_skills())
    raw_skills = doc.get("skills") if isinstance(doc.get("skills"), dict) else {}
    for key in SKILL_CATEGORIES:
        items = raw_skills.get(key) or []
        skills[key] = [dict(item) for item in items if isinstance(item, dict)]
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return {
        "userId": str(doc.get("userId") or user_id),
        "skills": skills,
        "recurringMistakes": [
            dict(item) for item in (doc.get("recurringMistakes") or []) if isinstance(item, dict)
        ],
        "strengths": [
            dict(item) for item in (doc.get("strengths") or []) if isinstance(item, dict)
        ],
        "improvementAreas": [
            dict(item) for item in (doc.get("improvementAreas") or []) if isinstance(item, dict)
        ],
        "learningPatterns": [
            dict(item) for item in (doc.get("learningPatterns") or []) if isinstance(item, dict)
        ],
        "overallAssessment": dict(doc.get("overallAssessment") or {}),
        "metadata": {
            "totalAnalyzedConversations": int(meta.get("totalAnalyzedConversations") or 0),
            "lastAnalyzedConversationId": meta.get("lastAnalyzedConversationId"),
            "lastAnalyzedAt": meta.get("lastAnalyzedAt"),
        },
        "version": int(doc.get("version") or 0),
        "createdAt": doc.get("createdAt"),
        "updatedAt": doc.get("updatedAt"),
    }


def validate_learning_output(
    raw: Any,
    *,
    min_confidence: float,
    pronunciation_evidence: bool,
) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = parse_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            return {"signals": [], "strengths": [], "patterns": [], "overallAssessment": {}}
    if not isinstance(raw, dict):
        return {"signals": [], "strengths": [], "patterns": [], "overallAssessment": {}}
    signals: list[dict[str, Any]] = []
    for item in raw.get("signals") or []:
        if not isinstance(item, dict):
            continue
        category = _category(item.get("category"))
        if category not in ALLOWED_CATEGORIES:
            continue
        if category == "pronunciation" and not pronunciation_evidence:
            continue
        skill = _skill_slug(item.get("skill"))
        issue = _trim(item.get("issue") or item.get("description"))
        if not skill or not issue:
            continue
        if _is_pollution(skill, issue):
            continue
        confidence = _clamp(item.get("confidence"))
        if confidence is None or confidence < min_confidence:
            continue
        signals.append(
            {
                "category": category,
                "skill": skill,
                "issue": issue,
                "severity": _severity(item.get("severity")),
                "confidence": confidence,
            }
        )
    strengths: list[dict[str, Any]] = []
    for item in raw.get("strengths") or []:
        if not isinstance(item, dict):
            continue
        skill = _skill_slug(item.get("skill"))
        description = _trim(item.get("description") or item.get("strength"))
        if not skill or not description:
            continue
        if _is_pollution(skill, description):
            continue
        if _TOPIC_POLLUTION.search(description):
            continue
        confidence = _clamp(item.get("confidence"))
        if confidence is None or confidence < min_confidence:
            continue
        strengths.append(
            {"skill": skill, "description": description, "confidence": confidence}
        )
    patterns: list[dict[str, Any]] = []
    for item in raw.get("patterns") or raw.get("learningPatterns") or []:
        if not isinstance(item, dict):
            continue
        description = _trim(item.get("description") or item.get("pattern"))
        if not description or _TOPIC_POLLUTION.search(description):
            continue
        if _is_pollution(description, description):
            continue
        confidence = _clamp(item.get("confidence"))
        if confidence is None or confidence < min_confidence:
            continue
        patterns.append({"description": description, "confidence": confidence})
    assessment = raw.get("overallAssessment") if isinstance(raw.get("overallAssessment"), dict) else {}
    return {
        "signals": signals,
        "strengths": strengths,
        "patterns": patterns,
        "overallAssessment": assessment,
    }


def _empty_learning_candidates() -> dict[str, Any]:
    return {"signals": [], "strengths": [], "patterns": [], "overallAssessment": {}}


def signals_from_summary(
    summary: dict[str, Any] | None,
    *,
    min_confidence: float,
    pronunciation_evidence: bool,
) -> dict[str, Any]:
    """Turn summary mistakes/strengths into learning signals when the LLM is empty."""
    if not summary:
        return _empty_learning_candidates()
    signals: list[dict[str, Any]] = []
    for item in summary.get("mistakes") or []:
        if not isinstance(item, dict):
            continue
        kind = _category(item.get("type") or item.get("category") or "grammar")
        if kind not in ALLOWED_CATEGORIES:
            kind = "grammar"
        if kind == "pronunciation" and not pronunciation_evidence:
            continue
        issue = _trim(item.get("userText") or item.get("issue") or item.get("explanation"))
        if not issue or _is_pollution(issue, issue):
            continue
        skill = _skill_slug(item.get("skill") or kind or "sentence_formation") or "sentence_formation"
        signals.append(
            {
                "category": kind,
                "skill": skill,
                "issue": issue,
                "severity": _severity(item.get("severity") or "medium"),
                "confidence": max(min_confidence, 0.86),
            }
        )
    for item in summary.get("grammarPatterns") or []:
        text = _trim(item if not isinstance(item, dict) else item.get("pattern") or item.get("description"))
        if not text or _is_pollution(text, text):
            continue
        signals.append(
            {
                "category": "grammar",
                "skill": _skill_slug(text) or "grammar_pattern",
                "issue": text,
                "severity": "medium",
                "confidence": max(min_confidence, 0.84),
            }
        )
    strengths: list[dict[str, Any]] = []
    for item in summary.get("strengths") or []:
        if isinstance(item, dict):
            description = _trim(item.get("description") or item.get("strength"))
            skill = _skill_slug(item.get("skill") or "speaking") or "speaking"
        else:
            description = _trim(item)
            skill = "speaking"
        if not description or _is_pollution(skill, description):
            continue
        strengths.append(
            {
                "skill": skill,
                "description": description,
                "confidence": max(min_confidence, 0.84),
            }
        )
    return {
        "signals": signals,
        "strengths": strengths,
        "patterns": [],
        "overallAssessment": {},
    }


def _append_source(ids: list[str], conversation_id: str, limit: int) -> list[str]:
    out: list[str] = []
    for item in list(ids) + [conversation_id]:
        text = str(item)
        if text not in out:
            out.append(text)
    if len(out) > limit:
        return out[-limit:]
    return out


def _find_mistake(
    mistakes: list[dict[str, Any]], category: str, skill: str
) -> dict[str, Any] | None:
    for row in mistakes:
        if _category(row.get("category")) == category and _skill_slug(row.get("skill")) == skill:
            return row
    return None


def _priority(frequency: int, severity: str) -> str:
    if severity == "high" and frequency >= 2:
        return "HIGH"
    if frequency >= 3:
        return "HIGH"
    if frequency >= 2:
        return "MEDIUM"
    return "LOW"


def _trim_list(items: list[dict[str, Any]], limit: int, *, key: str) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return items

    def sort_key(row: dict[str, Any]) -> tuple:
        status = _trim(row.get("status"))
        resolved = 0 if status == STATUS_RESOLVED else 1
        return (
            resolved,
            int(row.get("frequency") or 0),
            str(row.get("lastDetectedAt") or row.get("lastSeenAt") or ""),
        )

    ranked = sorted(items, key=sort_key, reverse=True)
    return ranked[:limit]


def _rebuild_skills(
    mistakes: list[dict[str, Any]],
    strengths: list[dict[str, Any]],
    *,
    max_per_category: int,
) -> dict[str, list[dict[str, Any]]]:
    skills = empty_skills()
    for row in mistakes:
        category = _category(row.get("category"))
        if category not in skills:
            continue
        skills[category].append(
            {
                "skill": row.get("skill"),
                "status": row.get("status"),
                "severity": row.get("severity"),
                "frequency": row.get("frequency"),
                "confidence": row.get("confidence"),
            }
        )
    for row in strengths:
        category = _category(row.get("skill"))
        if category not in skills:
            continue
        skills[category].append(
            {
                "skill": row.get("skill"),
                "status": "STRENGTH",
                "strength": row.get("description"),
                "frequency": row.get("frequency"),
                "confidence": row.get("confidence"),
            }
        )
    for key in SKILL_CATEGORIES:
        skills[key] = _trim_list(skills[key], max_per_category, key="frequency")
    return skills


def _rebuild_improvement_areas(
    mistakes: list[dict[str, Any]],
    *,
    min_recurring: int,
    limit: int,
) -> list[dict[str, Any]]:
    areas: list[dict[str, Any]] = []
    for row in mistakes:
        frequency = int(row.get("frequency") or 0)
        status = _trim(row.get("status"))
        if frequency < min_recurring or status not in {STATUS_ACTIVE, STATUS_IMPROVING}:
            continue
        areas.append(
            {
                "skill": row.get("skill"),
                "priority": _priority(frequency, _severity(row.get("severity"))),
                "reason": (
                    f"Recurring {row.get('category')} errors across {frequency} conversations"
                    if frequency >= min_recurring
                    else _trim(row.get("issue"))
                ),
            }
        )
    areas.sort(key=lambda item: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(item["priority"], 0), reverse=True)
    return areas[:limit]


def _smooth_assessment(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    old_weight: float,
    new_weight: float,
    session_level: str = "",
) -> dict[str, Any]:
    out = dict(existing or {})
    current_level = _trim(out.get("level")).upper()
    proposed_level = _trim(incoming.get("level")).upper()
    if current_level not in CEFR_LEVELS:
        if session_level in CEFR_LEVELS:
            out["level"] = session_level
        elif proposed_level in CEFR_LEVELS:
            out["level"] = proposed_level
    for key in ("confidence", "fluency", "accuracy", "vocabulary", "pronunciation"):
        incoming_score = _clamp(incoming.get(key))
        if incoming_score is None:
            continue
        previous = _clamp(out.get(key))
        if previous is None:
            out[key] = round(incoming_score, 4)
        else:
            out[key] = round(previous * old_weight + incoming_score * new_weight, 4)
    return out


def merge_learning_memory(
    existing: dict[str, Any] | None,
    candidates: dict[str, Any],
    *,
    conversation_id: str,
    now: datetime,
    config: LearningMemoryConfig,
    session_level: str = "",
) -> dict[str, Any]:
    mistakes = [
        dict(item)
        for item in ((existing or {}).get("recurringMistakes") or [])
        if isinstance(item, dict) and _skill_slug(item.get("skill"))
    ]
    seen_keys = {
        (_category(item.get("category")), _skill_slug(item.get("skill")))
        for item in candidates.get("signals") or []
    }
    for signal in candidates.get("signals") or []:
        category = signal["category"]
        skill = signal["skill"]
        found = _find_mistake(mistakes, category, skill)
        if found is None:
            mistakes.append(
                {
                    "id": f"{category}:{skill}",
                    "category": category,
                    "skill": skill,
                    "issue": signal["issue"],
                    "severity": signal["severity"],
                    "frequency": 1,
                    "confidence": signal["confidence"],
                    "firstDetectedAt": now,
                    "lastDetectedAt": now,
                    "status": STATUS_ACTIVE,
                    "absentStreak": 0,
                    "sourceConversationIds": [conversation_id],
                }
            )
            continue
        sources = [str(item) for item in (found.get("sourceConversationIds") or [])]
        if conversation_id in sources:
            continue
        found["frequency"] = int(found.get("frequency") or 0) + 1
        found["lastDetectedAt"] = now
        found["issue"] = signal["issue"] or found.get("issue")
        found["severity"] = signal["severity"]
        found["confidence"] = max(float(found.get("confidence") or 0), signal["confidence"])
        found["absentStreak"] = 0
        if _trim(found.get("status")) == STATUS_RESOLVED:
            found["status"] = STATUS_ACTIVE
        elif _trim(found.get("status")) == STATUS_IMPROVING:
            found["status"] = STATUS_ACTIVE
        found["sourceConversationIds"] = _append_source(
            sources, conversation_id, config.max_source_conversation_ids
        )

    for row in mistakes:
        key = (_category(row.get("category")), _skill_slug(row.get("skill")))
        if key in seen_keys:
            continue
        status = _trim(row.get("status")) or STATUS_ACTIVE
        if status == STATUS_RESOLVED:
            continue
        streak = int(row.get("absentStreak") or 0) + 1
        row["absentStreak"] = streak
        if streak >= config.resolution_evidence_count:
            row["status"] = STATUS_RESOLVED
        elif streak >= config.improvement_evidence_count:
            row["status"] = STATUS_IMPROVING

    mistakes = _trim_list(
        mistakes, config.max_recurring_mistakes, key="frequency"
    )

    strengths = [
        dict(item)
        for item in ((existing or {}).get("strengths") or [])
        if isinstance(item, dict)
    ]
    for item in candidates.get("strengths") or []:
        needle = _norm(item["description"])
        skill = item["skill"]
        found = None
        for row in strengths:
            if _skill_slug(row.get("skill")) == skill and _norm(
                row.get("description") or row.get("strength")
            ) == needle:
                found = row
                break
        if found is None:
            strengths.append(
                {
                    "skill": skill,
                    "description": item["description"],
                    "strength": item["description"],
                    "frequency": 1,
                    "confidence": item["confidence"],
                    "lastSeenAt": now,
                }
            )
            continue
        found["frequency"] = int(found.get("frequency") or 0) + 1
        found["confidence"] = max(float(found.get("confidence") or 0), item["confidence"])
        found["lastSeenAt"] = now
        found["description"] = item["description"]
        found["strength"] = item["description"]
    strengths = _trim_list(strengths, config.max_strengths, key="frequency")

    patterns = [
        dict(item)
        for item in ((existing or {}).get("learningPatterns") or [])
        if isinstance(item, dict)
    ]
    for item in candidates.get("patterns") or []:
        needle = _norm(item["description"])
        found = None
        for row in patterns:
            if _norm(row.get("pattern") or row.get("description")) == needle:
                found = row
                break
        if found is None:
            patterns.append(
                {
                    "pattern": item["description"],
                    "frequency": 1,
                    "confidence": item["confidence"],
                    "lastSeenAt": now,
                }
            )
            continue
        found["frequency"] = int(found.get("frequency") or 0) + 1
        found["confidence"] = max(float(found.get("confidence") or 0), item["confidence"])
        found["lastSeenAt"] = now
        found["pattern"] = item["description"]
    patterns = _trim_list(patterns, config.max_patterns, key="frequency")

    skills = _rebuild_skills(
        mistakes, strengths, max_per_category=config.max_skills_per_category
    )
    improvement = _rebuild_improvement_areas(
        mistakes,
        min_recurring=config.min_recurring_occurrences,
        limit=config.max_improvement_areas,
    )
    assessment = _smooth_assessment(
        (existing or {}).get("overallAssessment") or {},
        candidates.get("overallAssessment") or {},
        old_weight=config.assessment_old_weight,
        new_weight=config.assessment_new_weight,
        session_level=session_level,
    )
    previous_meta = (existing or {}).get("metadata") if isinstance((existing or {}).get("metadata"), dict) else {}
    total = int(previous_meta.get("totalAnalyzedConversations") or 0) + 1
    return {
        "skills": skills,
        "recurringMistakes": mistakes,
        "strengths": strengths,
        "improvementAreas": improvement,
        "learningPatterns": patterns,
        "overallAssessment": assessment,
        "metadata": {
            "totalAnalyzedConversations": total,
            "lastAnalyzedConversationId": conversation_id,
            "lastAnalyzedAt": now,
            "lastProcessedConversationId": conversation_id,
            "lastProcessedAt": now,
        },
    }


class LearningAnalyzer(Protocol):
    def analyze_json(self, *, system: str, user: str) -> dict[str, Any]: ...


class LearningMemoryRepo(Protocol):
    def find_conversation_session(self, conversation_id: str) -> dict[str, Any] | None: ...
    def find_conversation_summary(self, conversation_id: str) -> dict[str, Any] | None: ...
    def find_learning_memory(self, user_id: str) -> dict[str, Any] | None: ...
    def apply_learning_memory_from_conversation(
        self, user_id: str, conversation_id: str, fields: dict[str, Any]
    ) -> dict[str, Any] | None: ...


class LearningMemoryService:
    def __init__(
        self,
        repo: LearningMemoryRepo,
        *,
        analyzer: LearningAnalyzer | None = None,
        config: LearningMemoryConfig | dict | None = None,
    ) -> None:
        self.repo = repo
        self.analyzer = analyzer
        if isinstance(config, LearningMemoryConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = LearningMemoryConfig.from_mapping(config)
        else:
            self.config = DEFAULT_LEARNING_MEMORY_CONFIG

    def getLearningMemory(self, userId: str) -> dict[str, Any]:
        uid = _trim(userId)
        return public_learning_memory(self.repo.find_learning_memory(uid), user_id=uid)

    def analyzeAndUpdateLearningMemory(
        self,
        conversationId: str,
        *,
        userId: str | None = None,
    ) -> dict[str, Any]:
        cid = _trim(conversationId)
        if not cid:
            raise ConversationNotFound()
        owner = _trim(userId) or None
        session = self.repo.find_conversation_session(cid)
        if session is None:
            raise ConversationNotFound()
        session_user = _trim(session.get("userId"))
        if owner and session_user != owner:
            raise ConversationAccessDenied()
        if not session_user:
            raise ConversationAccessDenied()
        if session.get("status") != COMPLETED:
            raise ConversationNotCompleted()
        summary = self.repo.find_conversation_summary(cid)
        if summary is None or (
            summary.get("summaryStatus")
            and summary.get("summaryStatus") != COMPLETED
        ):
            raise SummaryNotFound()
        existing = self.repo.find_learning_memory(session_user)
        if cid in _processed_ids(existing):
            return public_learning_memory(existing, user_id=session_user)
        return self._analyze_and_store(session_user, cid, session, summary, existing)

    def _analyze_and_store(
        self,
        user_id: str,
        conversation_id: str,
        session: dict[str, Any],
        summary: dict[str, Any],
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        pronunciation_evidence = _has_pronunciation_evidence(summary)
        candidates = _empty_learning_candidates()
        llm_ok = False
        raw_proposed = False
        if self.analyzer is not None:
            prompt = build_learning_analysis_prompt(
                existing_memory=existing,
                summary=summary,
                pronunciation_evidence=pronunciation_evidence,
            )
            try:
                raw = self.analyzer.analyze_json(
                    system=LEARNING_ANALYSIS_SYSTEM_PROMPT,
                    user=prompt,
                )
                if isinstance(raw, dict):
                    raw_proposed = bool(raw.get("signals") or raw.get("strengths") or raw.get("patterns"))
                candidates = validate_learning_output(
                    raw,
                    min_confidence=self.config.min_confidence,
                    pronunciation_evidence=pronunciation_evidence,
                )
                llm_ok = True
            except Exception as exc:  # noqa: BLE001
                call_log.warn(
                    "LEARNING",
                    f"llm skip, using summary signals: {exc}",
                    extra={
                        "conversationId": conversation_id,
                        "userId": user_id,
                    },
                )
        if not candidates["signals"] and not candidates["strengths"] and not (llm_ok and raw_proposed):
            fallback = signals_from_summary(
                summary,
                min_confidence=self.config.min_confidence,
                pronunciation_evidence=pronunciation_evidence,
            )
            candidates = {
                **candidates,
                "signals": fallback["signals"],
                "strengths": fallback["strengths"],
            }
        if self.analyzer is None and not candidates["signals"] and not candidates["strengths"]:
            raise LearningMemoryUpdateFailed()
        now = _utc_now()
        session_level = _trim(
            session.get("languageLevelAtStart") or session.get("languageLevel")
        ).upper()
        merged = merge_learning_memory(
            existing,
            candidates,
            conversation_id=conversation_id,
            now=now,
            config=self.config,
            session_level=session_level,
        )
        fields = {
            **merged,
            "updatedAt": now,
        }
        saved = self.repo.apply_learning_memory_from_conversation(
            user_id, conversation_id, fields
        )
        if saved is None:
            current = self.repo.find_learning_memory(user_id)
            if current is not None:
                return public_learning_memory(current, user_id=user_id)
            raise LearningMemoryUpdateFailed()
        return public_learning_memory(saved, user_id=user_id)
