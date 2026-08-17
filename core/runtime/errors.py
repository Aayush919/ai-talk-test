"""LangGraph runtime errors — never used to mutate Mongo memory collections."""


class ConversationRuntimeError(Exception):
    code = "RUNTIME_ERROR"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class RuntimeStateInvalid(ConversationRuntimeError):
    code = "RUNTIME_STATE_INVALID"
