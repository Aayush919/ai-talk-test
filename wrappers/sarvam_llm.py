"""Sarvam LLM — Indian-English / Hinglish voice replies."""

from __future__ import annotations

from typing import Any

import httpx

from core import call_log
from core.keys import KeyPool
from core.prompts import build_system_prompt
from core.topics import Topic

SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
FALLBACK_MODEL = "sarvam-105b"


class SarvamCoach:
    def __init__(self, keys: KeyPool, model: str) -> None:
        self._keys = keys
        self.model = (model or "sarvam-105b-conversations").strip()
        self._http = httpx.Client(timeout=httpx.Timeout(18.0, connect=5.0))

    def reply(
        self,
        history: list[dict[str, str]],
        keywords: list[str],
        topic: Topic | None = None,
        memory_block: str = "",
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    keywords,
                    topic_title=topic.title if topic else "Free talk",
                    topic_prompt=(
                        topic.prompt
                        if topic
                        else "Have a natural English conversation and coach gently."
                    ),
                    memory_block=memory_block,
                ),
            },
            *history,
        ]
        trimmed = messages[:1] + messages[1:][-12:]

        def _once(api_key: str) -> str:
            try:
                return self._call(api_key, self.model, trimmed)
            except Exception as exc:  # noqa: BLE001
                if self.model != FALLBACK_MODEL:
                    call_log.warn(
                        "LLM",
                        "sarvam model fail — retrying flagship",
                        extra={"model": self.model, "fallback": FALLBACK_MODEL, "err": str(exc)[:160]},
                    )
                    return self._call(api_key, FALLBACK_MODEL, trimmed)
                raise

        return self._keys.run(_once)

    def _call(
        self, api_key: str, model: str, messages: list[dict[str, str]]
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 160,
            "n": 1,
            # Thinking burns tokens + latency on a live call
            "reasoning_effort": None,
        }
        try:
            resp = self._http.post(
                SARVAM_URL,
                headers={
                    "api-subscription-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if resp.status_code >= 400:
                # Some plans reject null reasoning_effort
                if resp.status_code == 400 and "reasoning" in (resp.text or "").lower():
                    payload.pop("reasoning_effort", None)
                    resp = self._http.post(
                        SARVAM_URL,
                        headers={
                            "api-subscription-key": api_key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
            if resp.status_code >= 400:
                raise RuntimeError(f"Sarvam {resp.status_code}: {resp.text[:240]}")
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            call_log.error("LLM", str(exc), extra={"model": model, "provider": "sarvam"})
            raise
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Sarvam returned no choices")
        msg = (choices[0].get("message") or {}).get("content") or ""
        return str(msg).strip()
