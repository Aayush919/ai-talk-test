"""Pick Groq or Sarvam coach; same reply() surface."""

from __future__ import annotations

import time
from typing import Protocol

from wrappers.groq_llm import GroqCoach
from wrappers.sarvam_llm import SarvamCoach


class CoachLLM(Protocol):
    def reply(
        self,
        history: list[dict[str, str]],
        keywords: list[str],
    ) -> str: ...

    def speak(self, *, system: str, user: str) -> str: ...

    def analyze_json(self, *, system: str, user: str) -> dict: ...


def _llm_kwargs(settings) -> dict:
    return {
        "live_timeout": float(getattr(settings, "live_llm_timeout", 5.0) or 5.0),
        "post_call_timeout": float(getattr(settings, "post_call_llm_timeout", 30.0) or 30.0),
        "live_max_tokens": int(getattr(settings, "live_llm_max_tokens", 160) or 160),
    }


class CoachWithFallback:
    """Try primary (Sarvam); on failure use Groq so the live call does not die."""

    def __init__(self, primary: CoachLLM, fallback: CoachLLM | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    def reply(
        self,
        history: list[dict[str, str]],
        keywords: list[str],
    ) -> str:
        t0 = time.perf_counter()
        try:
            return self.primary.reply(history, keywords)
        except Exception as exc:
            if self.fallback is None:
                raise
            from core import call_log

            call_log.warn(
                "LLM",
                "primary failed — using Groq fallback",
                extra={
                    "err_type": type(exc).__name__,
                    "err": str(exc)[:240],
                    "provider": type(self.primary).__name__,
                    "model": getattr(self.primary, "model", None),
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "attempt": 1,
                },
            )
            return self.fallback.reply(history, keywords)

    def speak(self, *, system: str, user: str) -> str:
        t0 = time.perf_counter()
        try:
            return self.primary.speak(system=system, user=user)
        except Exception as exc:
            if self.fallback is None:
                raise
            from core import call_log

            call_log.warn(
                "LLM",
                "primary speak failed — using Groq fallback",
                extra={
                    "err_type": type(exc).__name__,
                    "err": str(exc)[:240],
                    "provider": type(self.primary).__name__,
                    "model": getattr(self.primary, "model", None),
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "attempt": 1,
                },
            )
            return self.fallback.speak(system=system, user=user)

    def analyze_json(self, *, system: str, user: str) -> dict:
        t0 = time.perf_counter()
        try:
            return self.primary.analyze_json(system=system, user=user)
        except Exception as exc:
            if self.fallback is None:
                raise
            from core import call_log

            call_log.warn(
                "LLM",
                "primary analysis failed — using Groq fallback",
                extra={
                    "err_type": type(exc).__name__,
                    "err": str(exc)[:240],
                    "provider": type(self.primary).__name__,
                    "model": getattr(self.primary, "model", None),
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                    "attempt": 1,
                },
            )
            return self.fallback.analyze_json(system=system, user=user)


def build_llm(settings) -> CoachLLM:
    provider = (getattr(settings, "llm_provider", "groq") or "groq").lower()
    kwargs = _llm_kwargs(settings)
    groq = None
    if getattr(settings, "groq_keys", None) is not None:
        groq = GroqCoach(settings.groq_keys, settings.groq_model, **kwargs)
    if provider == "sarvam":
        sarvam = SarvamCoach(settings.sarvam_keys, settings.sarvam_model, **kwargs)
        return CoachWithFallback(sarvam, groq)
    if groq is None:
        raise RuntimeError("GROQ_API_KEYS missing")
    return groq
