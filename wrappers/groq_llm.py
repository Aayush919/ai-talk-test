"""Groq LLM — low-latency voice replies."""

from __future__ import annotations

from groq import Groq

from core import call_log
from core.keys import KeyPool
from core.prompts import build_system_prompt
from core.topics import Topic


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
                ),
            },
            *history,
        ]

        def _once(api_key: str) -> str:
            # Short context = faster TTFT
            trimmed = messages[:1] + messages[1:][-8:]
            kwargs: dict = {
                "model": self.model,
                "messages": trimmed,
                "temperature": 0.45,
                "max_completion_tokens": 72,
            }
            if "gpt-oss" in self.model:
                kwargs["reasoning_effort"] = "low"
            try:
                completion = self._client(api_key).chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                call_log.error("LLM", str(exc), extra={"model": self.model})
                raise
            msg = completion.choices[0].message
            return (getattr(msg, "content", None) or "").strip()

        return self._keys.run(_once)
