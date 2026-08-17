"""Topic progress errors — stable codes for the call handler."""


class TopicProgressError(Exception):
    code = "TOPIC_PROGRESS_ERROR"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class UserNotFound(TopicProgressError):
    code = "USER_NOT_FOUND"


class UserEnglishLevelRequired(TopicProgressError):
    code = "USER_ENGLISH_LEVEL_REQUIRED"


class TopicsNotFoundForLevel(TopicProgressError):
    code = "TOPICS_NOT_FOUND_FOR_LEVEL"


class TopicNotFound(TopicProgressError):
    code = "TOPIC_NOT_FOUND"


class SessionNotFound(TopicProgressError):
    code = "SESSION_NOT_FOUND"


class TopicProgressInternalError(TopicProgressError):
    code = "TOPIC_PROGRESS_INTERNAL_ERROR"


class TopicHasNoGoals(TopicProgressError):
    code = "TOPIC_HAS_NO_GOALS"


class TopicProgressUpdateFailed(TopicProgressError):
    code = "TOPIC_PROGRESS_UPDATE_FAILED"
