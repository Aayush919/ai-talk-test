"""Groq LLM — low-latency voice replies."""

from __future__ import annotations

import json

from groq import Groq

from core import call_log
from core.keys import KeyPool
from core.prompts import build_system_prompt


FALLBACK_MODEL = "llama-3.1-8b-instant"


def _is_harmony_glitch(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "tool_use_failed" in text or "but model called a tool" in text


def _failed_generation(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else {}
        text = str(err.get("failed_generation") or "")
        if text.strip():
            return text
    blob = str(exc)
    for marker in ("'failed_generation': '", '"failed_generation": "', "failed_generation': "):
        idx = blob.find(marker)
        if idx < 0:
            continue
        rest = blob[idx + len(marker) :]
        start = rest.find("{")
        if start < 0:
            continue
        return rest[start:]
    return ""


class GroqCoach:
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
        self.model = model
        self.live_timeout = float(live_timeout)
        self.post_call_timeout = float(post_call_timeout)
        self.live_max_tokens = int(live_max_tokens)
        self._clients: dict[str, Groq] = {}

    def _client(self, api_key: str) -> Groq:
        client = self._clients.get(api_key)
        if client is None:
            client = Groq(api_key=api_key, timeout=self.post_call_timeout)
            self._clients[api_key] = client
        return client

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

        def _once(api_key: str) -> str:
            # STM: last 6 turns (12 messages) + system prompt
            trimmed = messages[:1] + messages[1:][-12:]
            try:
                return self._call(api_key, self.model, trimmed, timeout=self.live_timeout)
            except Exception as exc:  # noqa: BLE001
                # gpt-oss sometimes emits harmony channel tokens that Groq reads
                # as a tool call; a live call must not die for that
                if _is_harmony_glitch(exc) and self.model != FALLBACK_MODEL:
                    call_log.warn(
                        "LLM",
                        "harmony glitch — retrying on fallback model",
                        extra={"model": self.model, "fallback": FALLBACK_MODEL},
                    )
                    return self._call(
                        api_key, FALLBACK_MODEL, trimmed, timeout=self.live_timeout
                    )
                raise

        return self._keys.run(_once)

    def speak(self, *, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _once(api_key: str) -> str:
            return self._call(
                api_key,
                self.model,
                messages,
                timeout=self.live_timeout,
                max_tokens=self.live_max_tokens,
                json_mode=False,
            )

        return self._keys.run(_once)

    def _call(
        self,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        *,
        timeout: float,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        tokens = max_tokens
        if tokens is None:
            tokens = 160 if "gpt-oss" in model else 80
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": 0.5 if not json_mode else 0.2,
            "max_completion_tokens": tokens,
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            completion = self._client(api_key).chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            failed = _failed_generation(exc)
            if failed:
                try:
                    return parse_json_object(failed)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            call_log.error(
                "LLM",
                str(exc),
                extra={"err_type": type(exc).__name__, "model": model, "provider": "groq"},
            )
            raise
        msg = completion.choices[0].message
        return (getattr(msg, "content", None) or "").strip()

    def analyze_json(self, *, system: str, user: str) -> dict:
        from core.conversations.summary_service import parse_json_object

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        def _once(api_key: str) -> dict:
            try:
                raw = self._call(
                    api_key,
                    self.model,
                    messages,
                    timeout=self.post_call_timeout,
                    max_tokens=4096,
                    json_mode=True,
                )
            except Exception:
                raw = self._call(
                    api_key,
                    self.model,
                    messages,
                    timeout=self.post_call_timeout,
                    max_tokens=4096,
                    json_mode=False,
                )
            return parse_json_object(raw)

        return self._keys.run(_once)
