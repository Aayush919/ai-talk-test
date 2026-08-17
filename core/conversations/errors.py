"""Conversation session / message errors — stable codes for the call handler."""


class ConversationError(Exception):
    code = "CONVERSATION_ERROR"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class ConversationNotFound(ConversationError):
    code = "CONVERSATION_NOT_FOUND"


class ConversationAccessDenied(ConversationError):
    code = "CONVERSATION_ACCESS_DENIED"


class ConversationNotActive(ConversationError):
    code = "CONVERSATION_NOT_ACTIVE"


class EmptyMessage(ConversationError):
    code = "EMPTY_MESSAGE"


class InvalidMessageRole(ConversationError):
    code = "INVALID_MESSAGE_ROLE"


class ConversationNotCompleted(ConversationError):
    code = "CONVERSATION_NOT_COMPLETED"


class InsufficientConversationData(ConversationError):
    code = "INSUFFICIENT_CONVERSATION_DATA"


class SummaryGenerationFailed(ConversationError):
    code = "SUMMARY_GENERATION_FAILED"


class SummaryNotFound(ConversationError):
    code = "SUMMARY_NOT_FOUND"
