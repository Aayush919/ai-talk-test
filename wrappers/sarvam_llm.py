"""Sarvam LLM — Indian-English / Hinglish voice replies."""

from __future__ import annotations

from typing import Any

import httpx

from core import call_log
from core.keys import KeyPool
from core.prompts import build_system_prompt

SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
FALLBACK_MODEL = "sarvam-105b"


class SarvamCoach:
    def __init__(
        self,
        keys: KeyPool,
        model: str,
        *,
        live_timeout: float = 5.0,
        post_call_timeout: float = 30.0,
        live_max_tokens: int = 160,
    ) -> None:
        self._keys = keys
        self.model = (model or "sarvam-105b-conversations").strip()
        self.live_timeout = float(live_timeout)
        self.post_call_timeout = float(post_call_timeout)
        self.live_max_tokens = int(live_max_tokens)
        self._http = httpx.Client(
            timeout=httpx.Timeout(self.post_call_timeout, connect=5.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=20),
        )

    def reply(
        self,
        history: list[dict[str, str]],
        keywords: list[str],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(keywords),
            },
            *history,
        ]
        trimmed = messages[:1] + messages[1:][-12:]

        def _once(api_key: str) -> str:
            try:
                return self._call(
                    api_key,
                    self.model,
                    trimmed,
                    max_tokens=self.live_max_tokens,
                    timeout=self.live_timeout,
                )
            except Exception as exc:  # noqa: BLE001
                if self.model != FALLBACK_MODEL:
                    call_log.warn(
                        "LLM",
                        "sarvam model fail — retrying flagship",
                        extra={"model": self.model, "fallback": FALLBACK_MODEL, "err": str(exc)[:160]},
                    )
                    return self._call(
                        api_key,
                        FALLBACK_MODEL,
                        trimmed,
                        max_tokens=self.live_max_tokens,
                        timeout=self.live_timeout,
                    )
                raise

        return self._keys.run(_once)

    def speak(self, *, system: str, user: str) -> str:
        """Live voice: plain text, no JSON mode, short timeout, thinking off."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _once(api_key: str) -> str:
            return self._call(
                api_key,
                self.model,
                messages,
                max_tokens=self.live_max_tokens,
                timeout=self.live_timeout,
            )

        return self._keys.run(_once)

    def _call(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        timeout: float,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": max_tokens,
            "n": 1,
            "reasoning_effort": None,
        }
        try:
            resp = self._post(api_key, payload, timeout)
            if resp.status_code >= 400:
                if resp.status_code == 400 and "reasoning" in (resp.text or "").lower():
                    payload.pop("reasoning_effort", None)
                    resp = self._post(api_key, payload, timeout)
            if resp.status_code >= 400:
                raise RuntimeError(f"Sarvam {resp.status_code}: {resp.text[:240]}")
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            call_log.error(
                "LLM",
                str(exc),
                extra={
                    "err_type": type(exc).__name__,
                    "model": model,
                    "provider": "sarvam",
                },
            )
            raise
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Sarvam returned no choices")
        msg = (choices[0].get("message") or {}).get("content") or ""
        return str(msg).strip()

    def _post(self, api_key: str, payload: dict[str, Any], timeout: float):
        return self._http.post(
            SARVAM_URL,
            headers={
                "api-subscription-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=httpx.Timeout(timeout, connect=5.0),
        )

    def analyze_json(self, *, system: str, user: str) -> dict:
        from core.conversations.summary_service import parse_json_object

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _once(api_key: str) -> dict:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 4096,
                "n": 1,
                "response_format": {"type": "json_object"},
                "reasoning_effort": None,
            }
            try:
                resp = self._post(api_key, payload, self.post_call_timeout)
                if resp.status_code >= 400:
                    text = (resp.text or "").lower()
                    if resp.status_code == 400 and "reasoning" in text:
                        payload.pop("reasoning_effort", None)
                    else:
                        payload.pop("response_format", None)
                    resp = self._post(api_key, payload, self.post_call_timeout)
                if resp.status_code >= 400:
                    raise RuntimeError(f"Sarvam {resp.status_code}: {resp.text[:240]}")
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("Sarvam returned no choices")
                msg = (choices[0].get("message") or {}).get("content") or ""
                return parse_json_object(str(msg).strip())
            except Exception as exc:  # noqa: BLE001
                call_log.error(
                    "LLM",
                    str(exc),
                    extra={
                        "err_type": type(exc).__name__,
                        "model": self.model,
                        "provider": "sarvam",
                    },
                )
                raise

        return self._keys.run(_once)
