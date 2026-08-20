"""Lightweight engagement and user-intent signals from the current utterance."""

from __future__ import annotations

import re
from typing import Any

_REFUSAL = re.compile(
    r"\b(i don't want to|don't want to talk|not comfortable|skip that|change (the )?topic)\b",
    re.I,
)
_CONFUSION = re.compile(
    r"\b(i don't know|i dont know|not sure|what does .+ mean|"
    r"i don't get|i dont get|don't get your point|dont get your point|"
    r"i don't understand|i dont understand|what happened)\b",
    re.I,
)
_CORRECTION_REQUEST = re.compile(
    r"\b(was (that|my english) correct|did i say (it|that) (right|correct)|correct my|how do i say)\b",
    re.I,
)
_GOODBYE = re.compile(r"\b(bye|goodbye|got to go|i have to go|that's all|end (the )?call)\b", re.I)
_TRANSLATE = re.compile(r"\b(what is the english (word|for)|how do you say|matlab kya)\b", re.I)
_REPEAT_COMPLAINT = re.compile(
    r"\bagain and again\b|\bwhy (?:are you )?(?:again|repeating)\b|\bstop repeating\b|"
    r"\balready told you\b|\bi already told\b|\bi already said\b|"
    r"\byou already (?:asked|told)\b|\bkeep repeating\b|\bsame question\b",
    re.I,
)
_MEMORY_PROBE = re.compile(
    r"\bdo you (?:know|remember)\b|\bdo you even know\b|"
    r"\byou know (?:what|my)\b|\bwhat (?:am i|did i|do i)\b|"
    r"\btell me (?:what|my)\b|\bremember (?:what|my)\b",
    re.I,
)
_ACK_WORDS = frozenset(
    """
    yeah yes yep yup ok okay hmm uh oh right sure good fine correct true
    that thats it's its
    """.split()
)
_KEEP_SHORT = frozenset(
    {
        "yoga",
        "shower",
        "cricket",
        "breakfast",
        "tea",
        "coffee",
        "office",
        "coding",
        "evening",
        "morning",
    }
)


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def detect_engagement(text: str) -> str:
    words = [part for part in _trim(text).split() if part]
    lower = _trim(text).lower()
    if not words:
        return "LOW"
    if len(words) <= 2 or lower in {"yes", "no", "ok", "okay", "yeah", "hmm"}:
        return "LOW"
    if "don't know" in lower or "dont know" in lower:
        return "LOW"
    if len(words) >= 18 or (len(words) >= 10 and "?" in text):
        return "HIGH"
    return "NORMAL"


def is_memory_probe(text: str) -> bool:
    return bool(_MEMORY_PROBE.search(_trim(text)))


def is_confused_turn(text: str) -> bool:
    return bool(_CONFUSION.search(_trim(text)))


def detect_user_intent(text: str) -> str:
    raw = _trim(text)
    if not raw:
        return "SILENCE"
    if _GOODBYE.search(raw):
        return "GOODBYE"
    if _CORRECTION_REQUEST.search(raw):
        return "CORRECTION_REQUEST"
    if is_memory_probe(raw):
        return "MEMORY_PROBE"
    if _REPEAT_COMPLAINT.search(raw):
        return "REPEAT_COMPLAINT"
    if is_confused_turn(raw):
        return "CONFUSION"
    if _TRANSLATE.search(raw):
        return "QUESTION"
    if _REFUSAL.search(raw):
        return "REFUSAL"
    if raw.endswith("?") or raw.lower().startswith(("what ", "why ", "how ", "where ", "when ")):
        return "QUESTION"
    words = raw.split()
    if len(words) <= 2:
        return "SMALL_TALK"
    return "ANSWER"


def is_low_content_turn(text: str) -> bool:
    """Yeah / okay / 'I I' — do not spend an LLM turn or ask a new question."""
    raw = text or ""
    if "?" in raw:
        return False
    cleaned = "".join(
        ch if ch.isalnum() or ch.isspace() else " "
        for ch in raw.lower().replace("'", "")
    )
    words = cleaned.split()
    if not words:
        return True
    joined = " ".join(words)
    if any(
        token in joined
        for token in (
            "remember",
            "do you know",
            "what am i",
            "what did i",
            "understand",
            "get your point",
            "hobby",
            "hobbies",
            "told you",
            "my name",
            "from",
        )
    ):
        return False
    if len(words) == 1 and words[0] not in _ACK_WORDS and words[0] not in _KEEP_SHORT:
        return True
    if all(word in _ACK_WORDS for word in words) and len(words) <= 6:
        return True
    if len(words) <= 2 and len(set(words)) == 1 and words[0] in _ACK_WORDS | {"i", "we"}:
        return True
    return False


def is_off_topic_question(text: str, topic_title: str = "") -> bool:
    if detect_user_intent(text) != "QUESTION":
        return False
    title = _trim(topic_title).lower()
    if not title:
        return False
    return title.split()[0] not in text.lower()
