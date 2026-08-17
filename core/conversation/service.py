"""AI conversation helpers used by the LangGraph live turn."""

from __future__ import annotations

from typing import Any

from core.conversation.config import ConversationConfig, DEFAULT_CONVERSATION_CONFIG
from core.conversation.correction import CorrectionService
from core.conversation.response import generateNextQuestion, parse_ai_response, spoken_text_only


class AIConversationService:
    """Conversation + correction policy. Does not write Mongo or Qdrant."""

    def __init__(
        self,
        *,
        config: ConversationConfig | None = None,
        corrections: CorrectionService | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONVERSATION_CONFIG
        self.corrections = corrections or CorrectionService(self.config)

    def generateNextQuestion(
        self, context: dict[str, Any], decision: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return generateNextQuestion(context, decision)

    def parse(self, raw: Any) -> dict[str, Any] | None:
        return parse_ai_response(raw)

    def user_facing(self, decision: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(decision or {})
        correction = data.get("correction") if isinstance(data.get("correction"), dict) else None
        if correction:
            correction = {
                "original": correction.get("original"),
                "corrected": correction.get("corrected"),
                "type": correction.get("type"),
                "explanation": correction.get("explanation"),
            }
        return {
            "text": spoken_text_only(data),
            "intent": data.get("intent") or "FOLLOW_UP",
            "question": data.get("question"),
            "correction": correction,
            "goalEvidence": data.get("goalEvidence"),
            "shouldContinue": bool(data.get("shouldContinue", True)),
            "shouldTransition": bool(data.get("shouldTransition", False)),
        }
