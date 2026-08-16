"""Pick Groq or Sarvam coach; same reply() surface."""

from __future__ import annotations

from typing import Protocol

from core.topics import Topic
from wrappers.groq_llm import GroqCoach
from wrappers.sarvam_llm import SarvamCoach


class CoachLLM(Protocol):
    def reply(
        self,
        history: list[dict[str, str]],
        keywords: list[str],
        topic: Topic | None = None,
        memory_block: str = "",
    ) -> str: ...


class CoachWithFallback:
    """Try primary (Sarvam); on failure use Groq so the live call does not die."""

    def __init__(self, primary: CoachLLM, fallback: CoachLLM | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    def reply(
        self,
        history: list[dict[str, str]],
        keywords: list[str],
        topic: Topic | None = None,
        memory_block: str = "",
    ) -> str:
        try:
            return self.primary.reply(history, keywords, topic=topic, memory_block=memory_block)
        except Exception:
            if self.fallback is None:
                raise
            from core import call_log

            call_log.warn("LLM", "primary failed — using Groq fallback")
            return self.fallback.reply(
                history, keywords, topic=topic, memory_block=memory_block
            )


def build_llm(settings) -> CoachLLM:
    provider = (getattr(settings, "llm_provider", "groq") or "groq").lower()
    groq = None
    if getattr(settings, "groq_keys", None) is not None:
        groq = GroqCoach(settings.groq_keys, settings.groq_model)
    if provider == "sarvam":
        sarvam = SarvamCoach(settings.sarvam_keys, settings.sarvam_model)
        return CoachWithFallback(sarvam, groq)
    if groq is None:
        raise RuntimeError("GROQ_API_KEYS missing")
    return groq
