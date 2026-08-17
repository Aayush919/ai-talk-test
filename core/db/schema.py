"""Mongo + Qdrant architecture lock. No conversation flow here.

`users` already exists elsewhere — this module never creates, indexes, or writes it.
Relation: userId → users._id (ID format comes later).
"""

from __future__ import annotations

# --- Mongo collections (users is owned outside this service) ---
USERS = "users"
TOPICS = "topics"
TOPIC_PROGRESS = "topic_progress"
CONVERSATION_SESSIONS = "conversation_sessions"
MESSAGES = "messages"
CONVERSATION_SUMMARIES = "conversation_summaries"
USER_PROFILE_MEMORY = "user_profile_memory"
LEARNING_MEMORY = "learning_memory"
MEMORY_METADATA = "memory_metadata"

OWNED_COLLECTIONS = (
    TOPICS,
    TOPIC_PROGRESS,
    CONVERSATION_SESSIONS,
    MESSAGES,
    CONVERSATION_SUMMARIES,
    USER_PROFILE_MEMORY,
    LEARNING_MEMORY,
    MEMORY_METADATA,
)

SESSION_STATUS = ("ACTIVE", "COMPLETED", "INTERRUPTED", "FAILED")
# Atlas already validates USER/ASSISTANT/SYSTEM; lowercase kept for local/dev.
MESSAGE_ROLES = ("USER", "ASSISTANT", "SYSTEM", "user", "assistant", "system")
TOPIC_PROGRESS_STATUS = ("NOT_STARTED", "IN_PROGRESS", "COMPLETED")
QDRANT_MEMORY_TYPES = (
    "PROFILE_FACT",
    "LEARNING_PATTERN",
    "LEARNING_WEAKNESS",
    "LEARNING_STRENGTH",
    "EXPERIENCE",
    "PREFERENCE",
    "CONVERSATION_MEMORY",
)

# Qdrant — one collection, payload-filtered by tenantId + userId
QDRANT_COLLECTION_DEFAULT = "english_coach_memories"
QDRANT_VECTOR_SIZE = 384  # default; override with EMBEDDING_DIMENSION


def _str_or_id() -> dict:
    """userId / topicId / conversationId — ObjectId or string until IDs are locked."""
    return {"bsonType": ["objectId", "string"]}


def _opt_id() -> dict:
    return {"bsonType": ["objectId", "string", "null"]}


def _date() -> dict:
    return {"bsonType": ["date", "null"]}


def _num() -> dict:
    return {"bsonType": ["int", "double", "long", "null"]}


COLLECTION_VALIDATORS: dict[str, dict] = {
    TOPICS: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["title", "slug"],
            "properties": {
                "title": {"bsonType": "string"},
                "slug": {"bsonType": "string"},
                "description": {"bsonType": ["string", "null"]},
                "order": _num(),
                "level": {"enum": ["A1", "A2", "B1", "B2", "C1"]},
                "goals": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "object",
                        "required": ["key", "description"],
                        "properties": {
                            "key": {"bsonType": "string"},
                            "description": {"bsonType": "string"},
                        },
                    },
                },
                "completionCriteria": {
                    "bsonType": "object",
                    "properties": {
                        "minimumGoals": _num(),
                        "minimumConversationSeconds": _num(),
                    },
                },
                "isActive": {"bsonType": ["bool", "null"]},
                "createdAt": _date(),
                "updatedAt": _date(),
            },
        }
    },
    TOPIC_PROGRESS: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["userId", "topicId"],
            "properties": {
                "userId": _str_or_id(),
                "topicId": _str_or_id(),
                "status": {"bsonType": ["string", "null"]},
                "progress": _num(),
                "goalsCompleted": {"bsonType": ["array", "null"]},
                "goalsRemaining": {"bsonType": ["array", "null"]},
                "attemptCount": _num(),
                "lastConversationId": _opt_id(),
                "processedConversationIds": {"bsonType": ["array", "null"]},
                "lastProcessedConversationId": _opt_id(),
                "lastProcessedAt": _date(),
                "startedAt": _date(),
                "completedAt": _date(),
                "updatedAt": _date(),
                "needsRevisit": {"bsonType": ["bool", "null"]},
                "engineEvaluatedConversationId": _opt_id(),
                "currentGoalId": {"bsonType": ["string", "null"]},
            },
        }
    },
    CONVERSATION_SESSIONS: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["userId", "topicId"],
            "properties": {
                "userId": _str_or_id(),
                "topicId": _str_or_id(),
                "status": {"bsonType": ["string", "null"]},
                "callType": {"bsonType": ["string", "null"]},
                "startedAt": _date(),
                "endedAt": _date(),
                "durationSeconds": _num(),
                "duration": _num(),
                "languageLevelAtStart": {"bsonType": ["string", "null"]},
                "languageLevelAtEnd": {"bsonType": ["string", "null"]},
                "messageCount": _num(),
                "lastMessageAt": _date(),
                "endReason": {"bsonType": ["string", "null"]},
                "createdAt": _date(),
                "updatedAt": _date(),
            },
        }
    },
    MESSAGES: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["conversationId", "role", "content", "sequence"],
            "properties": {
                "conversationId": _str_or_id(),
                "userId": _str_or_id(),
                "topicId": _str_or_id(),
                "role": {"enum": list(MESSAGE_ROLES)},
                "content": {"bsonType": "string"},
                "sequence": _num(),
                "timestamp": _date(),
                "metadata": {"bsonType": ["object", "null"]},
                "createdAt": _date(),
                "updatedAt": _date(),
            },
        }
    },
    CONVERSATION_SUMMARIES: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["conversationId", "userId"],
            "properties": {
                "conversationId": _str_or_id(),
                "userId": _str_or_id(),
                "topicId": _str_or_id(),
                "summaryStatus": {"bsonType": ["string", "null"]},
                "summary": {"bsonType": ["string", "null"]},
                "keyPoints": {"bsonType": ["array", "null"]},
                "goals": {"bsonType": ["array", "null"]},
                "goalsCovered": {"bsonType": ["array", "null"]},
                "goalsMissed": {"bsonType": ["array", "null"]},
                "mistakes": {"bsonType": ["array", "null"]},
                "corrections": {"bsonType": ["array", "null"]},
                "strengths": {"bsonType": ["array", "null"]},
                "weaknesses": {"bsonType": ["array", "null"]},
                "importantFacts": {"bsonType": ["array", "null"]},
                "vocabulary": {"bsonType": ["array", "null"]},
                "grammarPatterns": {"bsonType": ["array", "null"]},
                "fluencyObservations": {"bsonType": ["array", "null"]},
                "conversationMetrics": {"bsonType": ["object", "null"]},
                "conversationQuality": {"bsonType": ["string", "double", "int", "null"]},
                "topicProgressAfterCall": {"bsonType": ["object", "int", "double", "null"]},
                "createdAt": _date(),
                "updatedAt": _date(),
            },
        }
    },
    USER_PROFILE_MEMORY: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["userId", "key", "value"],
            "properties": {
                "userId": _str_or_id(),
                "key": {"bsonType": "string"},
                "value": {
                    "bsonType": ["string", "int", "double", "bool", "object", "array"]
                },
                "sourceConversationId": _str_or_id(),
                "confidence": _num(),
                "importance": _num(),
                "profile": {
                    "bsonType": ["object", "null"],
                    "properties": {
                        "name": {"bsonType": ["string", "null"]},
                        "profession": {"bsonType": ["string", "null"]},
                        "education": {"bsonType": ["string", "null"]},
                        "experience": {"bsonType": ["string", "null"]},
                        "location": {"bsonType": ["string", "null"]},
                        "interests": {"bsonType": ["array", "null"]},
                        "hobbies": {"bsonType": ["array", "null"]},
                        "goals": {"bsonType": ["array", "null"]},
                        "nativeLanguage": {"bsonType": ["string", "null"]},
                        "englishLearningGoal": {"bsonType": ["string", "null"]},
                        "preferredLearningStyle": {"bsonType": ["string", "null"]},
                        "communicationPreferences": {"bsonType": ["array", "null"]},
                    },
                },
                "facts": {
                    "bsonType": ["array", "null"],
                    "items": {
                        "bsonType": "object",
                        "required": ["key", "value"],
                        "properties": {
                            "key": {"bsonType": "string"},
                            "value": {
                                "bsonType": [
                                    "string",
                                    "int",
                                    "double",
                                    "bool",
                                    "object",
                                    "array",
                                ]
                            },
                            "confidence": _num(),
                            "sourceConversationId": _str_or_id(),
                            "firstSeenAt": _date(),
                            "lastConfirmedAt": _date(),
                        },
                    },
                },
                "version": _num(),
                "memoryStatus": {"bsonType": ["string", "null"]},
                "processedConversationIds": {"bsonType": ["array", "null"]},
                "lastProcessedConversationId": _str_or_id(),
                "lastProcessedAt": _date(),
                "profileMemoryMetadata": {"bsonType": ["object", "null"]},
                "createdAt": _date(),
                "updatedAt": _date(),
            },
        }
    },
    LEARNING_MEMORY: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["userId", "type", "value"],
            "properties": {
                "userId": _str_or_id(),
                "type": {"bsonType": "string"},
                "value": {"bsonType": "string"},
                "severity": {"bsonType": ["string", "null"]},
                "confidence": _num(),
                "sourceConversationId": _str_or_id(),
                "topicId": _str_or_id(),
                "frequency": _num(),
                "lastObservedAt": _date(),
                "skills": {"bsonType": ["object", "null"]},
                "recurringMistakes": {"bsonType": ["array", "null"]},
                "strengths": {"bsonType": ["array", "null"]},
                "improvementAreas": {"bsonType": ["array", "null"]},
                "learningPatterns": {"bsonType": ["array", "null"]},
                "overallAssessment": {"bsonType": ["object", "null"]},
                "metadata": {"bsonType": ["object", "null"]},
                "processedConversationIds": {"bsonType": ["array", "null"]},
                "version": _num(),
                "createdAt": _date(),
                "updatedAt": _date(),
            },
        }
    },
    MEMORY_METADATA: {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["userId", "tenantId", "qdrantPointId"],
            "properties": {
                "userId": _str_or_id(),
                "tenantId": {"bsonType": ["string", "null"]},
                "memoryId": _str_or_id(),
                "identityKey": {"bsonType": ["string", "null"]},
                "memoryType": {"bsonType": ["string", "null"]},
                "category": {"bsonType": ["string", "null"]},
                "skill": {"bsonType": ["string", "null"]},
                "content": {"bsonType": ["string", "null"]},
                "sourceType": {"bsonType": ["string", "null"]},
                "sourceId": _str_or_id(),
                "topicId": _str_or_id(),
                "topicLevel": {"bsonType": ["string", "null"]},
                "scope": {"bsonType": ["string", "null"]},
                "status": {"bsonType": ["string", "null"]},
                "importance": _num(),
                "confidence": _num(),
                "frequency": _num(),
                "qdrantPointId": {"bsonType": "string"},
                "embeddingModel": {"bsonType": ["string", "null"]},
                "embeddingVersion": {"bsonType": ["string", "null"]},
                "indexed": {"bsonType": ["bool", "null"]},
                "indexedAt": _date(),
                "lastSeenAt": _date(),
                "lastIndexedSourceId": _str_or_id(),
                "createdAt": _date(),
                "updatedAt": _date(),
            },
        }
    },
}

# (collection, keys, unique)
INDEXES: list[tuple[str, list[tuple[str, int]], bool]] = [
    (TOPICS, [("slug", 1)], True),
    (TOPICS, [("level", 1), ("order", 1)], True),
    (TOPICS, [("level", 1), ("isActive", 1), ("order", 1)], False),
    (TOPICS, [("isActive", 1)], False),
    (TOPIC_PROGRESS, [("userId", 1), ("topicId", 1)], True),
    (TOPIC_PROGRESS, [("userId", 1), ("status", 1)], False),
    (TOPIC_PROGRESS, [("lastConversationId", 1)], False),
    (CONVERSATION_SESSIONS, [("userId", 1)], False),
    (CONVERSATION_SESSIONS, [("userId", 1), ("status", 1)], False),
    (CONVERSATION_SESSIONS, [("userId", 1), ("startedAt", -1)], False),
    (CONVERSATION_SESSIONS, [("topicId", 1)], False),
    (CONVERSATION_SESSIONS, [("status", 1)], False),
    (MESSAGES, [("conversationId", 1), ("sequence", 1)], True),
    (MESSAGES, [("userId", 1), ("createdAt", -1)], False),
    (CONVERSATION_SUMMARIES, [("conversationId", 1)], True),
    (CONVERSATION_SUMMARIES, [("userId", 1)], False),
    (CONVERSATION_SUMMARIES, [("topicId", 1)], False),
    (USER_PROFILE_MEMORY, [("userId", 1)], True),
    (LEARNING_MEMORY, [("userId", 1)], True),
    (MEMORY_METADATA, [("qdrantPointId", 1)], True),
    (MEMORY_METADATA, [("memoryId", 1)], True),
    (MEMORY_METADATA, [("tenantId", 1), ("userId", 1), ("identityKey", 1)], True),
    (MEMORY_METADATA, [("userId", 1), ("memoryType", 1)], False),
    (MEMORY_METADATA, [("sourceId", 1)], False),
    (MEMORY_METADATA, [("indexed", 1)], False),
]
