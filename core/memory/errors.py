"""User profile memory errors — stable codes for the API / jobs."""


class ProfileMemoryError(Exception):
    code = "PROFILE_MEMORY_ERROR"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class ProfileAccessDenied(ProfileMemoryError):
    code = "ACCESS_DENIED"


class ProfileMemoryUpdateFailed(ProfileMemoryError):
    code = "PROFILE_MEMORY_UPDATE_FAILED"


class LearningMemoryError(Exception):
    code = "LEARNING_MEMORY_ERROR"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class LearningMemoryUpdateFailed(LearningMemoryError):
    code = "LEARNING_MEMORY_UPDATE_FAILED"
