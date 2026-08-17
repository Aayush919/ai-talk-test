"""Stable user profile memory — LLM proposes facts; backend decides what is stored."""

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
from core.memory.errors import ProfileAccessDenied, ProfileMemoryUpdateFailed
from core.memory.profile_prompt import (
    PROFILE_EXTRACTION_SYSTEM_PROMPT,
    build_profile_extraction_prompt,
)

COMPLETED = "COMPLETED"
MEMORY_COMPLETED = "COMPLETED"
MIN_CONFIDENCE = 0.75
REPLACE_CONFIDENCE = 0.90

ALLOWED_KEYS = frozenset(
    {
        "name",
        "profession",
        "education",
        "experience",
        "location",
        "interest",
        "hobby",
        "goal",
        "nativeLanguage",
        "englishLearningGoal",
        "preferredLearningStyle",
        "communicationPreference",
    }
)
ARRAY_KEY_TO_FIELD = {
    "interest": "interests",
    "hobby": "hobbies",
    "goal": "goals",
    "communicationPreference": "communicationPreferences",
}
SCALAR_KEY_TO_FIELD = {
    "name": "name",
    "profession": "profession",
    "education": "education",
    "experience": "experience",
    "location": "location",
    "nativeLanguage": "nativeLanguage",
    "englishLearningGoal": "englishLearningGoal",
    "preferredLearningStyle": "preferredLearningStyle",
}
ARRAY_FIELDS = (
    "interests",
    "hobbies",
    "goals",
    "communicationPreferences",
)
_POLLUTION = re.compile(
    r"\b("
    r"tired|mood|grammar|mistake|pronunciation|"
    r"topic progress|goals? completed|last answer|"
    r"article mistakes?|present perfect"
    r")\b",
    re.I,
)
_FRACTION = re.compile(r"\d+\s*/\s*\d+")
_GOAL_TO_KEY = {
    "name": "name",
    "location": "location",
    "education_or_work": "profession",
    "work": "profession",
    "profession": "profession",
    "hobbies": "hobby",
    "hobby": "hobby",
    "future_goal": "goal",
    "goal": "goal",
}
_NAME_RE = re.compile(
    r"(?:learner'?s\s+name\s+is|my\s+name\s+is|name\s+is)\s+"
    r"([A-Za-z][A-Za-z'.-]*(?:\s+[A-Za-z][A-Za-z'.-]*){0,2})",
    re.I,
)
_LOCATION_RE = re.compile(
    r"(?:live[s]?\s+in|living\s+in|based\s+in)\s+([A-Za-z][A-Za-z\s.'-]{1,40})",
    re.I,
)
_PROFESSION_RE = re.compile(
    r"(?:i\s+am\s+a[n]?|i'?m\s+a[n]?|is\s+a[n]?|works?\s+as(?:\s+a[n]?)?)\s+"
    r"([^.,:;]{3,50})",
    re.I,
)
_HOBBY_RE = re.compile(
    r"(?:hobby\s+is|hobbies\s+are|i\s+(?:like|love|enjoy)\s+(?:playing\s+)?)"
    r"([^.,:;]{3,40})",
    re.I,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm_compare(value: Any) -> str:
    return " ".join(_trim(value).lower().split())


def _normalize_array_item(value: str) -> str:
    return _norm_compare(value)


def _array_contains(items: list[str], candidate: str) -> bool:
    needle = _norm_compare(candidate)
    return any(_norm_compare(item) == needle for item in items)


def _is_pollution(key: str, value: str) -> bool:
    blob = f"{key} {value}"
    if _FRACTION.search(blob):
        return True
    return _POLLUTION.search(blob) is not None


def _account_display_name(user: dict[str, Any] | None) -> str:
    if not user:
        return ""
    for field in ("displayName", "name", "fullName"):
        text = _trim(user.get(field))
        if text:
            return text
    return ""


def _values(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [_trim(item) for item in raw if _trim(item)]
    text = _trim(raw)
    return [text] if text else []


def _important_facts(summary: dict[str, Any]) -> list[Any]:
    facts = summary.get("importantFacts")
    return facts if isinstance(facts, list) else []


def _clean_capture(value: str) -> str:
    text = _trim(value).strip(" .,'\"")
    text = re.split(r"\s+(?:but|and|who|which|that)\s+", text, maxsplit=1)[0]
    return _trim(text)


def _candidate(key: str, value: str, confidence: float = 0.92) -> dict[str, Any] | None:
    text = _clean_capture(value)
    if not text or _is_pollution(key, text):
        return None
    if key == "name" and " " not in text and len(text) < 2:
        return None
    return {"key": key, "value": text, "confidence": confidence}


def _facts_from_text(text: str, confidence: float) -> list[dict[str, Any]]:
    blob = _trim(text)
    if len(blob) < 8:
        return []
    out: list[dict[str, Any]] = []
    name = _NAME_RE.search(blob)
    if name:
        row = _candidate("name", name.group(1), confidence)
        if row:
            out.append(row)
    loc = _LOCATION_RE.search(blob)
    if loc:
        row = _candidate("location", loc.group(1), confidence)
        if row:
            out.append(row)
    job = _PROFESSION_RE.search(blob)
    if job:
        row = _candidate("profession", job.group(1), confidence)
        if row:
            out.append(row)
    hobby = _HOBBY_RE.search(blob)
    if hobby:
        row = _candidate("hobby", hobby.group(1), confidence)
        if row:
            out.append(row)
    return out


def candidates_from_summary(summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Stable profile facts already present in the conversation summary."""
    if not summary:
        return []
    found: dict[str, dict[str, Any]] = {}
    for item in _important_facts(summary):
        if isinstance(item, dict):
            text = _trim(item.get("fact"))
            try:
                conf = float(item.get("confidence") if item.get("confidence") is not None else 0.9)
            except (TypeError, ValueError):
                conf = 0.9
        else:
            text = _trim(item)
            conf = 0.9
        if conf < MIN_CONFIDENCE:
            continue
        for row in _facts_from_text(text, max(conf, 0.9)):
            found[row["key"]] = row
    for goal in summary.get("goals") or []:
        if not isinstance(goal, dict):
            continue
        if _trim(goal.get("status")).upper() not in {"COMPLETED", "PARTIAL"}:
            continue
        key = _GOAL_TO_KEY.get(_trim(goal.get("goalId") or goal.get("key")))
        evidence = _trim(goal.get("evidence"))
        if not key or len(evidence) < 10:
            continue
        parsed = _facts_from_text(evidence, 0.93)
        matched = [row for row in parsed if row["key"] == key]
        row = matched[0] if matched else _candidate(key, evidence, 0.9)
        if row:
            found.setdefault(row["key"], row)
    return list(found.values())


def _merge_candidates(
    base: list[dict[str, Any]], extra: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {row["key"]: row for row in base}
    for row in extra:
        current = by_key.get(row["key"])
        if current is None or float(row["confidence"]) >= float(current["confidence"]):
            by_key[row["key"]] = row
    return list(by_key.values())


def _processed_ids(doc: dict[str, Any] | None) -> set[str]:
    if not doc:
        return set()
    ids = {str(item) for item in (doc.get("processedConversationIds") or [])}
    last = _trim(doc.get("lastProcessedConversationId"))
    if last:
        ids.add(last)
    meta = doc.get("profileMemoryMetadata")
    if isinstance(meta, dict):
        last_meta = _trim(meta.get("lastProcessedConversationId"))
        if last_meta:
            ids.add(last_meta)
    return ids


def public_profile(doc: dict[str, Any] | None, *, user_id: str = "") -> dict[str, Any]:
    if not doc:
        return {
            "userId": user_id,
            "profile": {},
            "facts": [],
            "version": 0,
            "memoryStatus": "PENDING",
        }
    profile = dict(doc.get("profile") or {})
    return {
        "userId": str(doc.get("userId") or user_id),
        "profile": profile,
        "facts": [dict(item) for item in (doc.get("facts") or []) if isinstance(item, dict)],
        "version": int(doc.get("version") or 0),
        "memoryStatus": _trim(doc.get("memoryStatus")) or MEMORY_COMPLETED,
        "lastProcessedConversationId": doc.get("lastProcessedConversationId"),
        "lastProcessedAt": doc.get("lastProcessedAt"),
        "createdAt": doc.get("createdAt"),
        "updatedAt": doc.get("updatedAt"),
    }


def validate_candidates(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = parse_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(raw, dict):
        return []
    facts = raw.get("facts")
    if not isinstance(facts, list):
        return []
    out: list[dict[str, Any]] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        key = _trim(item.get("key"))
        action = _trim(item.get("action")).upper() or "UPSERT"
        if action == "IGNORE" or action != "UPSERT":
            continue
        if key not in ALLOWED_KEYS:
            continue
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            continue
        if confidence < MIN_CONFIDENCE:
            continue
        for value in _values(item.get("value")):
            if _is_pollution(key, value):
                continue
            out.append({"key": key, "value": value, "confidence": confidence})
    return out


def _copy_profile(existing: dict[str, Any] | None) -> dict[str, Any]:
    src = dict((existing or {}).get("profile") or {})
    profile: dict[str, Any] = {}
    for key, value in src.items():
        if key in ARRAY_FIELDS:
            profile[key] = [item for item in (value or []) if _trim(item)]
        elif _trim(value):
            profile[key] = _trim(value)
    for field in ARRAY_FIELDS:
        profile.setdefault(field, [])
    return profile


def _upsert_fact(
    facts: list[dict[str, Any]],
    *,
    key: str,
    value: str,
    confidence: float,
    conversation_id: str,
    now: datetime,
    replace_same_key: bool,
) -> list[dict[str, Any]]:
    needle = _norm_compare(value)
    for fact in facts:
        if _trim(fact.get("key")) != key:
            continue
        same_value = _norm_compare(fact.get("value")) == needle
        if same_value:
            fact["lastConfirmedAt"] = now
            fact["confidence"] = max(float(fact.get("confidence") or 0), confidence)
            return facts
        if replace_same_key:
            fact["value"] = value
            fact["confidence"] = confidence
            fact["sourceConversationId"] = conversation_id
            fact["lastConfirmedAt"] = now
            return facts
    facts.append(
        {
            "key": key,
            "value": value,
            "confidence": confidence,
            "sourceConversationId": conversation_id,
            "firstSeenAt": now,
            "lastConfirmedAt": now,
        }
    )
    return facts


def merge_profile(
    existing: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    *,
    conversation_id: str,
    now: datetime,
    account_user: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = _copy_profile(existing)
    facts = [
        dict(item)
        for item in ((existing or {}).get("facts") or [])
        if isinstance(item, dict) and _trim(item.get("key"))
    ]
    account_name = _account_display_name(account_user)

    for cand in candidates:
        key = cand["key"]
        value = cand["value"]
        confidence = float(cand["confidence"])
        if key in ARRAY_KEY_TO_FIELD:
            field = ARRAY_KEY_TO_FIELD[key]
            stored = _normalize_array_item(value)
            if not stored:
                continue
            current = list(profile.get(field) or [])
            if not _array_contains(current, stored):
                current.append(stored)
            profile[field] = current
            facts = _upsert_fact(
                facts,
                key=key,
                value=stored,
                confidence=confidence,
                conversation_id=conversation_id,
                now=now,
                replace_same_key=False,
            )
            continue

        field = SCALAR_KEY_TO_FIELD.get(key)
        if not field:
            continue
        old = _trim(profile.get(field))
        if old and _norm_compare(old) == _norm_compare(value):
            facts = _upsert_fact(
                facts,
                key=key,
                value=old,
                confidence=confidence,
                conversation_id=conversation_id,
                now=now,
                replace_same_key=True,
            )
            continue
        if old and confidence < REPLACE_CONFIDENCE:
            continue
        if key == "name" and account_name and confidence < REPLACE_CONFIDENCE:
            if _norm_compare(value) != _norm_compare(account_name):
                continue
        if key == "profession" and old and _norm_compare(old) != _norm_compare(value):
            facts = _upsert_fact(
                facts,
                key="previous_profession",
                value=old,
                confidence=max(confidence, REPLACE_CONFIDENCE),
                conversation_id=conversation_id,
                now=now,
                replace_same_key=True,
            )
        profile[field] = value
        facts = _upsert_fact(
            facts,
            key=key,
            value=value,
            confidence=confidence,
            conversation_id=conversation_id,
            now=now,
            replace_same_key=True,
        )

    for field in ARRAY_FIELDS:
        seen: list[str] = []
        for item in profile.get(field) or []:
            stored = _normalize_array_item(str(item))
            if stored and not _array_contains(seen, stored):
                seen.append(stored)
        if seen:
            profile[field] = seen
        else:
            profile.pop(field, None)
    return profile, facts


class ProfileAnalyzer(Protocol):
    def analyze_json(self, *, system: str, user: str) -> dict[str, Any]: ...


class UserProfileMemoryRepo(Protocol):
    def find_conversation_session(self, conversation_id: str) -> dict[str, Any] | None: ...
    def find_conversation_summary(self, conversation_id: str) -> dict[str, Any] | None: ...
    def find_user_profile(self, user_id: str) -> dict[str, Any] | None: ...
    def apply_profile_from_conversation(
        self, user_id: str, conversation_id: str, fields: dict[str, Any]
    ) -> dict[str, Any] | None: ...
    def find_user(self, user_id: str) -> dict[str, Any] | None: ...


class UserProfileMemoryService:
    def __init__(
        self,
        repo: UserProfileMemoryRepo,
        *,
        analyzer: ProfileAnalyzer | None = None,
    ) -> None:
        self.repo = repo
        self.analyzer = analyzer

    def getUserProfileMemory(
        self,
        userId: str,
        *,
        requesterId: str | None = None,
    ) -> dict[str, Any]:
        uid = _trim(userId)
        if not uid:
            raise ProfileAccessDenied()
        requester = _trim(requesterId)
        if requester and requester != uid:
            raise ProfileAccessDenied()
        return public_profile(self.repo.find_user_profile(uid), user_id=uid)

    def extractAndUpdateUserProfileMemory(
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
        existing = self.repo.find_user_profile(session_user)
        if cid in _processed_ids(existing):
            return public_profile(existing, user_id=session_user)
        return self._extract_and_store(session_user, cid, summary, existing)

    def _extract_and_store(
        self,
        user_id: str,
        conversation_id: str,
        summary: dict[str, Any],
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        candidates = candidates_from_summary(summary)
        if self.analyzer is not None:
            prompt = build_profile_extraction_prompt(
                existing_profile=(existing or {}).get("profile") if existing else {},
                summary=_trim(summary.get("summary")),
                important_facts=_important_facts(summary),
            )
            try:
                raw = self.analyzer.analyze_json(
                    system=PROFILE_EXTRACTION_SYSTEM_PROMPT,
                    user=prompt,
                )
                candidates = _merge_candidates(candidates, validate_candidates(raw))
            except Exception as exc:  # noqa: BLE001
                call_log.warn(
                    "PROFILE",
                    f"llm skip, using summary facts: {exc}",
                    extra={
                        "conversationId": conversation_id,
                        "userId": user_id,
                    },
                )
                if not candidates:
                    raise ProfileMemoryUpdateFailed() from exc
        elif not candidates:
            raise ProfileMemoryUpdateFailed()
        account_user = None
        find_user = getattr(self.repo, "find_user", None)
        if callable(find_user):
            try:
                account_user = find_user(user_id)
            except Exception:
                account_user = None
        now = _utc_now()
        profile, facts = merge_profile(
            existing,
            candidates,
            conversation_id=conversation_id,
            now=now,
            account_user=account_user,
        )
        fields = {
            "profile": profile,
            "facts": facts,
            "memoryStatus": MEMORY_COMPLETED,
            "lastProcessedConversationId": conversation_id,
            "lastProcessedAt": now,
            "updatedAt": now,
            "profileMemoryMetadata": {
                "lastProcessedConversationId": conversation_id,
                "lastProcessedAt": now,
            },
        }
        saved = self.repo.apply_profile_from_conversation(
            user_id, conversation_id, fields
        )
        if saved is None:
            current = self.repo.find_user_profile(user_id)
            if current is not None:
                return public_profile(current, user_id=user_id)
            raise ProfileMemoryUpdateFailed()
        return public_profile(saved, user_id=user_id)
