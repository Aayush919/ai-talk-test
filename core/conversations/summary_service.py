"""Structured analysis of a COMPLETED conversation — does not touch topic_progress."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Protocol

from core import call_log
from core.conversations.errors import (
    ConversationAccessDenied,
    ConversationError,
    ConversationNotCompleted,
    ConversationNotFound,
    InsufficientConversationData,
    SummaryGenerationFailed,
    SummaryNotFound,
)
from core.conversations.summary_prompt import (
    ANALYSIS_SYSTEM_PROMPT,
    build_analysis_user_prompt,
)
from core.topics.errors import TopicProgressInternalError

COMPLETED = "COMPLETED"
FAILED = "FAILED"
SUMMARY_PENDING = "PENDING"
SUMMARY_GENERATING = "GENERATING"
SUMMARY_COMPLETED = "COMPLETED"
SUMMARY_FAILED = "FAILED"
_GOAL_STATUSES = frozenset({"COMPLETED", "PARTIAL", "NOT_ATTEMPTED"})
_MISTAKE_TYPES = frozenset(
    {"GRAMMAR", "VOCABULARY", "PRONUNCIATION", "FLUENCY", "OTHER"}
)
_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "as",
        "at",
        "by",
        "from",
        "that",
        "this",
        "it",
        "user",
        "learner",
        "name",
        "my",
        "i",
        "he",
        "she",
        "they",
        "his",
        "her",
        "their",
    }
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _words(text: str) -> list[str]:
    return [part for part in (text or "").split() if part]


def parse_json_object(text: str) -> dict[str, Any]:
    raw = _trim(text)
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("analysis JSON must be an object")
    return data


def conversation_metrics(messages: list[dict[str, Any]]) -> dict[str, int]:
    user_count = 0
    assistant_count = 0
    user_words = 0
    assistant_words = 0
    for row in messages:
        role = _trim(row.get("role")).lower()
        count = len(_words(_trim(row.get("content"))))
        if role == "user":
            user_count += 1
            user_words += count
        elif role == "assistant":
            assistant_count += 1
            assistant_words += count
    return {
        "userMessageCount": user_count,
        "assistantMessageCount": assistant_count,
        "estimatedUserWords": user_words,
        "estimatedAssistantWords": assistant_words,
    }


def _transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in messages:
        role = _trim(row.get("role")).lower() or "unknown"
        content = _trim(row.get("content"))
        if not content:
            continue
        seq = row.get("sequence")
        prefix = f"[{seq}] " if seq is not None else ""
        lines.append(f"{prefix}{role}: {content}")
    return "\n".join(lines)


def _meaningful_user_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in messages
        if _trim(row.get("role")).lower() == "user" and _trim(row.get("content"))
    ]


def _corpus(messages: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for row in messages:
        tokens.update(re.findall(r"[a-z0-9']+", _trim(row.get("content")).lower()))
    return tokens


def _fact_grounded(fact: str, corpus: set[str]) -> bool:
    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9']+", _trim(fact).lower())
        if tok not in _STOP and len(tok) > 2
    ]
    if not tokens:
        return False
    missing = [tok for tok in tokens if tok not in corpus]
    return (len(missing) / len(tokens)) <= 0.34


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _trim(item)
        if text:
            out.append(text)
    return out


def _topic_goals(topic: dict[str, Any] | None) -> list[dict[str, Any]]:
    goals = (topic or {}).get("goals") or []
    if not isinstance(goals, list):
        return []
    out: list[dict[str, Any]] = []
    for goal in goals:
        if isinstance(goal, dict) and _trim(goal.get("key")):
            out.append(
                {
                    "key": _trim(goal.get("key")),
                    "description": _trim(goal.get("description")),
                }
            )
        elif isinstance(goal, str) and _trim(goal):
            out.append({"key": _trim(goal), "description": _trim(goal)})
    return out


def public_summary(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    if not doc:
        return None
    cid = doc.get("conversationId")
    return {
        "summaryId": str(doc.get("_id")) if doc.get("_id") is not None else None,
        "conversationId": str(cid) if cid is not None else None,
        "userId": doc.get("userId"),
        "topicId": str(doc["topicId"]) if doc.get("topicId") is not None else None,
        "summaryStatus": doc.get("summaryStatus") or SUMMARY_COMPLETED,
        "summary": doc.get("summary"),
        "keyPoints": doc.get("keyPoints") or [],
        "goals": doc.get("goals") or [],
        "mistakes": doc.get("mistakes") or [],
        "corrections": doc.get("corrections") or [],
        "strengths": doc.get("strengths") or [],
        "weaknesses": doc.get("weaknesses") or [],
        "importantFacts": doc.get("importantFacts") or [],
        "vocabulary": doc.get("vocabulary") or [],
        "grammarPatterns": doc.get("grammarPatterns") or [],
        "fluencyObservations": doc.get("fluencyObservations") or [],
        "conversationMetrics": doc.get("conversationMetrics") or {},
        "createdAt": doc.get("createdAt"),
        "updatedAt": doc.get("updatedAt"),
    }


class ConversationAnalyzer(Protocol):
    def analyze_json(self, *, system: str, user: str) -> dict[str, Any]: ...


class ConversationSummaryRepo(Protocol):
    def find_conversation_session(self, conversation_id: str) -> dict[str, Any] | None: ...
    def find_topic(self, topic_id: Any) -> dict[str, Any] | None: ...
    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]: ...
    def find_conversation_summary(self, conversation_id: str) -> dict[str, Any] | None: ...
    def upsert_conversation_summary(
        self, conversation_id: str, doc: dict[str, Any]
    ) -> dict[str, Any]: ...


class ConversationSummaryService:
    def __init__(
        self,
        repo: ConversationSummaryRepo,
        *,
        analyzer: ConversationAnalyzer | None = None,
    ) -> None:
        self.repo = repo
        self.analyzer = analyzer

    def generateConversationSummary(
        self,
        conversationId: str,
        *,
        userId: str | None = None,
    ) -> dict[str, Any]:
        cid = _trim(conversationId)
        if not cid:
            raise ConversationNotFound()
        owner = _trim(userId) or None
        try:
            session = self.repo.find_conversation_session(cid)
            self._assert_readable(session, owner)
            if session.get("status") != COMPLETED:
                raise ConversationNotCompleted()
            existing = self.repo.find_conversation_summary(cid)
            if existing and existing.get("summaryStatus") == SUMMARY_COMPLETED:
                out = public_summary(existing)
                if out:
                    return out
            messages = self.repo.list_messages(cid)
            if not _meaningful_user_messages(messages):
                raise InsufficientConversationData()
            metrics = conversation_metrics(messages)
            topic = self.repo.find_topic(session.get("topicId"))
            goals = _topic_goals(topic)
            if self.analyzer is None:
                raise SummaryGenerationFailed()
            raw = self.analyzer.analyze_json(
                system=ANALYSIS_SYSTEM_PROMPT,
                user=build_analysis_user_prompt(
                    topic=topic,
                    goals=goals,
                    transcript=_transcript(messages),
                ),
            )
            if isinstance(raw, str):
                raw = parse_json_object(raw)
            if not isinstance(raw, dict):
                raise SummaryGenerationFailed()
            analysis = self._normalize_analysis(
                raw, topic_goals=goals, metrics=metrics, messages=messages
            )
            existing = self.repo.find_conversation_summary(cid)
            if existing and existing.get("summaryStatus") == SUMMARY_COMPLETED:
                out = public_summary(existing)
                if out:
                    return out
            now = _utc_now()
            saved = self.repo.upsert_conversation_summary(
                cid,
                {
                    "conversationId": cid,
                    "userId": session.get("userId"),
                    "topicId": session.get("topicId"),
                    "summaryStatus": SUMMARY_COMPLETED,
                    **analysis,
                    "updatedAt": now,
                },
            )
            out = public_summary(saved)
            if not out:
                raise SummaryGenerationFailed()
            call_log.info(
                "SUMMARY",
                "conversation summary completed",
                extra={
                    "conversationId": cid,
                    "userId": session.get("userId"),
                    "topicId": str(session.get("topicId") or ""),
                    "summaryStatus": SUMMARY_COMPLETED,
                },
            )
            return out
        except ConversationError:
            raise
        except Exception as exc:  # noqa: BLE001
            call_log.error(
                "SUMMARY",
                f"generation failed: {exc}",
                extra={
                    "conversationId": cid,
                    "userId": owner,
                    "role": "summary",
                },
            )
            try:
                session = self.repo.find_conversation_session(cid)
                existing = self.repo.find_conversation_summary(cid)
                if existing and existing.get("summaryStatus") == SUMMARY_COMPLETED:
                    raise SummaryGenerationFailed() from exc
                if session and session.get("status") == COMPLETED:
                    self.repo.upsert_conversation_summary(
                        cid,
                        {
                            "conversationId": cid,
                            "userId": session.get("userId"),
                            "topicId": session.get("topicId"),
                            "summaryStatus": SUMMARY_FAILED,
                            "summary": None,
                            "updatedAt": _utc_now(),
                        },
                    )
            except Exception:
                pass
            if isinstance(exc, (json.JSONDecodeError, ValueError, TypeError)):
                raise SummaryGenerationFailed() from exc
            raise SummaryGenerationFailed() from exc

    def getConversationSummary(
        self, conversationId: str, userId: str
    ) -> dict[str, Any]:
        cid = _trim(conversationId)
        uid = _trim(userId)
        if not cid:
            raise ConversationNotFound()
        if not uid:
            raise ConversationAccessDenied()
        session = self.repo.find_conversation_session(cid)
        self._assert_readable(session, uid)
        doc = self.repo.find_conversation_summary(cid)
        if doc is None:
            raise SummaryNotFound()
        out = public_summary(doc)
        if not out:
            raise SummaryNotFound()
        return out

    def _assert_readable(
        self, session: dict[str, Any] | None, owner: str | None
    ) -> None:
        if session is None:
            raise ConversationNotFound()
        if owner and str(session.get("userId") or "") != owner:
            raise ConversationAccessDenied()

    def _normalize_analysis(
        self,
        raw: dict[str, Any],
        *,
        topic_goals: list[dict[str, Any]],
        metrics: dict[str, int],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summary = _trim(raw.get("summary"))
        if not summary:
            raise SummaryGenerationFailed()
        corpus = _corpus(messages)
        facts: list[dict[str, Any]] = []
        for item in raw.get("importantFacts") or []:
            if not isinstance(item, dict):
                continue
            fact = _trim(item.get("fact"))
            if not fact or not _fact_grounded(fact, corpus):
                continue
            try:
                confidence = float(item.get("confidence") if item.get("confidence") is not None else 0.5)
            except (TypeError, ValueError):
                confidence = 0.5
            facts.append(
                {
                    "fact": fact,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        return {
            "summary": summary,
            "keyPoints": _as_str_list(raw.get("keyPoints")),
            "goals": self._normalize_goals(raw.get("goals"), topic_goals),
            "mistakes": self._normalize_mistakes(raw.get("mistakes")),
            "corrections": self._normalize_corrections(raw.get("corrections")),
            "strengths": _as_str_list(raw.get("strengths")),
            "weaknesses": _as_str_list(raw.get("weaknesses")),
            "importantFacts": facts,
            "vocabulary": self._normalize_vocabulary(raw.get("vocabulary")),
            "grammarPatterns": _as_str_list(raw.get("grammarPatterns")),
            "fluencyObservations": _as_str_list(raw.get("fluencyObservations")),
            "conversationMetrics": metrics,
        }

    def _normalize_goals(
        self, raw_goals: Any, topic_goals: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        if isinstance(raw_goals, list):
            for item in raw_goals:
                if not isinstance(item, dict):
                    continue
                goal_id = _trim(item.get("goalId") or item.get("key"))
                status = _trim(item.get("status")).upper()
                if not goal_id or status not in _GOAL_STATUSES:
                    raise SummaryGenerationFailed()
                by_id[goal_id] = {
                    "goalId": goal_id,
                    "status": status,
                    "evidence": _trim(item.get("evidence")),
                }
        ordered: list[dict[str, Any]] = []
        seen: set[str] = set()
        for goal in topic_goals:
            key = goal["key"]
            seen.add(key)
            ordered.append(
                by_id.get(
                    key,
                    {
                        "goalId": key,
                        "status": "NOT_ATTEMPTED",
                        "evidence": "",
                    },
                )
            )
        for goal_id, row in by_id.items():
            if goal_id not in seen:
                ordered.append(row)
        return ordered

    def _normalize_mistakes(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            kind = _trim(item.get("type")).upper()
            user_text = _trim(item.get("userText"))
            if kind not in _MISTAKE_TYPES or not user_text:
                continue
            out.append(
                {
                    "type": kind,
                    "userText": user_text,
                    "correction": _trim(item.get("correction")),
                    "explanation": _trim(item.get("explanation")),
                }
            )
        return out

    def _normalize_corrections(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            original = _trim(item.get("original"))
            corrected = _trim(item.get("corrected"))
            if not original or not corrected:
                continue
            out.append(
                {
                    "original": original,
                    "corrected": corrected,
                    "category": _trim(item.get("category")),
                }
            )
        return out

    def _normalize_vocabulary(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            word = _trim(item.get("word"))
            if not word:
                continue
            out.append(
                {
                    "word": word,
                    "meaning": _trim(item.get("meaning")),
                    "context": _trim(item.get("context")),
                }
            )
        return out


def run_summary_job(
    service: ConversationSummaryService | None,
    conversation_id: str,
    *,
    user_id: str | None = None,
    progress_service: Any = None,
    profile_service: Any = None,
    learning_service: Any = None,
    semantic_service: Any = None,
) -> None:
    """Best-effort background hook. Never changes session COMPLETED → FAILED."""
    if service is None or not conversation_id:
        return
    try:
        result = service.generateConversationSummary(conversation_id, userId=user_id)
    except (InsufficientConversationData, ConversationNotCompleted) as exc:
        call_log.info(
            "SUMMARY",
            exc.code,
            extra={"conversationId": conversation_id, "userId": user_id},
        )
        return
    except ConversationError as exc:
        call_log.warn(
            "SUMMARY",
            exc.code,
            extra={"conversationId": conversation_id, "userId": user_id},
        )
        return
    except Exception as exc:  # noqa: BLE001
        call_log.error(
            "SUMMARY",
            f"job skip: {exc}",
            extra={"conversationId": conversation_id, "userId": user_id},
        )
        return
    if (result or {}).get("summaryStatus") != "COMPLETED":
        return
    if progress_service is not None:
        try:
            progress_service.updateTopicProgressFromSummary(
                conversation_id, userId=user_id
            )
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None) or "TOPIC_PROGRESS_UPDATE_FAILED"
            call_log.error(
                "PROGRESS",
                code,
                extra={
                    "conversationId": conversation_id,
                    "userId": user_id,
                    "detail": str(exc)[:240],
                },
            )
        try:
            evaluate = getattr(progress_service, "evaluateAfterConversation", None)
            if callable(evaluate):
                evaluate(conversation_id, userId=user_id)
        except Exception as exc:  # noqa: BLE001
            call_log.error(
                "TOPIC",
                getattr(exc, "code", None) or "TOPIC_ENGINE_EVALUATE_FAILED",
                extra={
                    "conversationId": conversation_id,
                    "userId": user_id,
                    "detail": str(exc)[:240],
                },
            )
    if profile_service is not None:
        try:
            profile_service.extractAndUpdateUserProfileMemory(
                conversation_id, userId=user_id
            )
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None) or "PROFILE_MEMORY_UPDATE_FAILED"
            call_log.error(
                "PROFILE",
                code,
                extra={
                    "conversationId": conversation_id,
                    "userId": user_id,
                    "detail": str(exc)[:240],
                },
            )
    if learning_service is not None:
        try:
            learning_service.analyzeAndUpdateLearningMemory(
                conversation_id, userId=user_id
            )
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "code", None) or "LEARNING_MEMORY_UPDATE_FAILED"
            call_log.error(
                "LEARNING",
                code,
                extra={
                    "conversationId": conversation_id,
                    "userId": user_id,
                    "detail": str(exc)[:240],
                },
            )
    if semantic_service is None:
        return
    try:
        semantic_service.extractAndIndexFromConversation(
            conversation_id, userId=user_id
        )
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "code", None) or "SEMANTIC_MEMORY_INDEX_FAILED"
        call_log.error(
            "MEMORY",
            code,
            extra={
                "conversationId": conversation_id,
                "userId": user_id,
                "detail": str(exc)[:240],
            },
        )
