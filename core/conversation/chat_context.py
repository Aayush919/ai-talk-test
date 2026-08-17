"""LiveKit-style ChatContext for the current call only.

Same idea as LiveKit Agents ChatContext: append this session's turns and
send them to the LLM. Not the LiveKit Agents runtime.

LangGraph still owns the checkpoint. Mongo and Qdrant are unchanged.
"""

from __future__ import annotations

from typing import Any

# ~15 min of short voice turns. Prefill stays small vs TTS/LLM.
MAX_CALL_ITEMS = 100
MAX_CALL_CHARS = 5500
MAX_COVERED = 16


def _trim(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm_role(role: str) -> str:
    raw = _trim(role).lower()
    if raw in {"assistant", "ai", "coach", "model"}:
        return "assistant"
    return "user"


class CallChatContext:
    """In-call message buffer. Drop oldest text first; keep Q→A covered notes."""

    def __init__(self, items: list[dict[str, str]] | None = None) -> None:
        self.items: list[dict[str, str]] = []
        for item in items or []:
            self.append(item.get("role") or "user", item.get("content") or "")

    def append(self, role: str, content: str) -> None:
        text = _trim(content)
        if not text:
            return
        self.items.append({"role": _norm_role(role), "content": text})
        if len(self.items) > MAX_CALL_ITEMS:
            self.items = self.items[-MAX_CALL_ITEMS:]

    @classmethod
    def from_runtime(
        cls,
        messages: list[Any] | None,
        *,
        current_user: str = "",
    ) -> "CallChatContext":
        ctx = cls()
        for item in messages or []:
            if not isinstance(item, dict):
                continue
            ctx.append(str(item.get("role") or "user"), str(item.get("content") or ""))
        extra = _trim(current_user)
        if extra:
            last = ctx.items[-1] if ctx.items else None
            if not last or last["role"] != "user" or last["content"] != extra:
                ctx.append("user", extra)
        return ctx

    def covered_answers(self) -> list[str]:
        covered: list[str] = []
        pending_q = ""
        for item in self.items:
            role = item["role"]
            text = item["content"]
            if role == "assistant" and "?" in text:
                pending_q = text.split("?")[0].strip() + "?"
                if len(pending_q) > 80:
                    pending_q = pending_q[:77] + "..."
                continue
            if role == "user" and pending_q and len(text.split()) >= 2:
                covered.append(f"{pending_q} → {text[:80]}")
                pending_q = ""
        return covered[-MAX_COVERED:]

    def for_llm(self) -> tuple[list[dict[str, str]], list[str]]:
        """Newest turns that fit the char budget, plus Q→A notes from the whole call."""
        covered = self.covered_answers()
        if not self.items:
            return [], covered
        kept: list[dict[str, str]] = []
        used = 0
        for item in reversed(self.items):
            size = len(item["content"]) + 8
            if kept and used + size > MAX_CALL_CHARS:
                break
            kept.append(item)
            used += size
        kept.reverse()
        return kept, covered
