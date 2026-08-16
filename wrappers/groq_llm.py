"""Groq LLM — low-latency voice replies."""

from __future__ import annotations

from groq import Groq

from core import call_log
from core.keys import KeyPool
from core.prompts import build_system_prompt
from core.topics import Topic


FALLBACK_MODEL = "llama-3.1-8b-instant"


def _is_harmony_glitch(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "tool_use_failed" in text or "but model called a tool" in text


class GroqCoach:
    def __init__(self, keys: KeyPool, model: str) -> None:
        self._keys = keys
        self.model = model
        self._clients: dict[str, Groq] = {}

    def _client(self, api_key: str) -> Groq:
        client = self._clients.get(api_key)
        if client is None:
            client = Groq(api_key=api_key)
            self._clients[api_key] = client
        return client

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

        def _once(api_key: str) -> str:
            # STM: last 6 turns (12 messages) + system memory card
            trimmed = messages[:1] + messages[1:][-12:]
            try:
                return self._call(api_key, self.model, trimmed)
            except Exception as exc:  # noqa: BLE001
                # gpt-oss sometimes emits harmony channel tokens that Groq reads
                # as a tool call; a live call must not die for that
                if _is_harmony_glitch(exc) and self.model != FALLBACK_MODEL:
                    call_log.warn(
                        "LLM",
                        "harmony glitch — retrying on fallback model",
                        extra={"model": self.model, "fallback": FALLBACK_MODEL},
                    )
                    return self._call(api_key, FALLBACK_MODEL, trimmed)
                raise

        return self._keys.run(_once)

    def _call(
        self, api_key: str, model: str, messages: list[dict[str, str]]
    ) -> str:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": 0.5,
            "max_completion_tokens": 160 if "gpt-oss" in model else 80,
        }
        if "gpt-oss" in model:
            kwargs["reasoning_effort"] = "low"
        try:
            completion = self._client(api_key).chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            call_log.error("LLM", str(exc), extra={"model": model})
            raise
        msg = completion.choices[0].message
        return (getattr(msg, "content", None) or "").strip()
