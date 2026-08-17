"""Serializable LangGraph runtime state. No DB connections, audio, or streams."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.runtime.errors import RuntimeStateInvalid

CorrectionMode = Literal["NONE", "SUBTLE", "NORMAL", "DIRECT", "IMMEDIATE", "DEFERRED"]
FollowUpStyle = Literal["OPEN", "SPECIFIC", "EXAMPLE_BASED"]
FocusType = Literal[
    "conversation", "grammar", "vocabulary", "fluency", "pronunciation"
]
GoalProgress = Literal["NOT_STARTED", "IN_PROGRESS", "GOOD"]
DecisionIntent = Literal[
    "OPEN_TOPIC",
    "FOLLOW_UP",
    "DEEPEN",
    "TRANSITION",
    "CLARIFY",
    "ENCOURAGE",
    "CLOSE",
    "ANSWER",
    "CORRECTION",
    "CLARIFICATION",
    "OFF_TOPIC",
    "CLOSING",
]

INTENTS = frozenset(
    {
        "OPEN_TOPIC",
        "FOLLOW_UP",
        "DEEPEN",
        "TRANSITION",
        "CLARIFY",
        "ENCOURAGE",
        "CLOSE",
        "ANSWER",
        "CORRECTION",
        "CLARIFICATION",
        "OFF_TOPIC",
        "CLOSING",
    }
)
FORBIDDEN_DECISION_KEYS = frozenset(
    {
        "databaseOperation",
        "topicProgress",
        "userId",
        "conversationId",
        "goalsCompleted",
        "goalsRemaining",
    }
)


class RuntimeMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class RuntimeProfileFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: str


class RuntimeLearningSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: str
    status: str
    category: str | None = None


class CurrentFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: FocusType = "conversation"
    skill: str | None = None


class CoachingStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctionMode: CorrectionMode = "SUBTLE"
    followUpStyle: FollowUpStyle = "OPEN"
    targetSkill: str | None = None


class UserContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profileFacts: list[RuntimeProfileFact] = Field(default_factory=list)
    learningSignals: list[RuntimeLearningSignal] = Field(default_factory=list)


class TurnOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goalProgress: GoalProgress | None = None
    needsFollowUp: bool | None = None
    detectedLearningSignal: bool | None = None


class ConversationDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response: str
    intent: DecisionIntent = "FOLLOW_UP"
    targetGoalId: str | None = None
    followUpNeeded: bool = True
    correction: dict[str, Any] | None = None


class ConversationRuntimeState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    userId: str
    conversationId: str
    topicId: str
    topicTitle: str = ""
    topicLevel: str | None = None
    topicStatus: str | None = None
    topicProgress: int | None = None
    currentGoalId: str | None = None
    currentGoalIndex: int | None = None
    goalsCompleted: list[str] = Field(default_factory=list)
    goalsRemaining: list[str] = Field(default_factory=list)
    recentMessages: list[RuntimeMessage] = Field(default_factory=list)
    lastUserMessage: str | None = None
    lastAssistantMessage: str | None = None
    conversationTurn: int = 0
    currentFocus: CurrentFocus | None = None
    coachingStrategy: CoachingStrategy = Field(default_factory=CoachingStrategy)
    userContext: UserContext = Field(default_factory=UserContext)
    pendingMemorySignals: list[dict[str, Any]] = Field(default_factory=list)
    relevantMemories: list[dict[str, Any]] = Field(default_factory=list)
    memoryRetrievalKey: str | None = None
    topicPlan: dict[str, Any] | None = None
    shouldContinue: bool = True
    turnOutcome: TurnOutcome | None = None
    conversationPhase: str | None = None
    userEngagement: str | None = None
    lastAssistantQuestion: str | None = None
    lastUserAnswer: str | None = None
    pendingGoalEvidence: dict[str, Any] | None = None


class RuntimeStateDict(TypedDict, total=False):
    userId: str
    conversationId: str
    topicId: str
    topicTitle: str
    topicLevel: str | None
    topicStatus: str | None
    topicProgress: int | None
    currentGoalId: str | None
    currentGoalIndex: int | None
    goalsCompleted: list[str]
    goalsRemaining: list[str]
    topicGoals: list[dict[str, str]]
    recentMessages: list[dict[str, str]]
    lastUserMessage: str | None
    lastAssistantMessage: str | None
    conversationTurn: int
    currentFocus: dict[str, Any] | None
    coachingStrategy: dict[str, Any]
    userContext: dict[str, Any]
    pendingMemorySignals: list[dict[str, Any]]
    relevantMemories: list[dict[str, Any]]
    memoryRetrievalKey: str
    topicPlan: dict[str, Any]
    shouldContinue: bool
    turnOutcome: dict[str, Any] | None
    lastDecision: dict[str, Any] | None
    runtimeMode: str
    incomingUserMessage: str | None
    incomingAssistantMessage: str | None
    contextRelevantMemories: list[dict[str, Any]]
    contextMemoryRetrievalKey: str
    contextSession: dict[str, Any] | None
    contextTopic: dict[str, Any] | None
    contextProgress: dict[str, Any] | None
    contextProfile: dict[str, Any] | None
    contextLearning: dict[str, Any] | None
    contextTopicPlan: dict[str, Any] | None
    conversationPhase: str | None
    userEngagement: str | None
    lastUserIntent: str | None
    lastUserAnswer: str | None
    lastAssistantQuestion: str | None
    lastMentionedEntities: list[str]
    recentQuestions: list[str]
    recentQuestionTypes: list[str]
    correctionState: dict[str, Any]
    pendingGoalEvidence: dict[str, Any] | None
    sttConfidence: float | None
    incomingSttConfidence: float | None
    pronunciationEvidence: bool


PUBLIC_KEYS = (
    "userId",
    "conversationId",
    "topicId",
    "topicTitle",
    "topicLevel",
    "topicStatus",
    "topicProgress",
    "currentGoalId",
    "currentGoalIndex",
    "goalsCompleted",
    "goalsRemaining",
    "recentMessages",
    "lastUserMessage",
    "lastAssistantMessage",
    "conversationTurn",
    "currentFocus",
    "coachingStrategy",
    "userContext",
    "pendingMemorySignals",
    "relevantMemories",
    "topicPlan",
    "shouldContinue",
    "turnOutcome",
    "conversationPhase",
    "userEngagement",
    "lastAssistantQuestion",
    "lastUserAnswer",
    "pendingGoalEvidence",
)


def validate_runtime_state(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        model = ConversationRuntimeState.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeStateInvalid() from exc
    return model.model_dump()


def public_runtime_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(raw or {})
    out = {key: data.get(key) for key in PUBLIC_KEYS}
    out.setdefault("goalsCompleted", [])
    out.setdefault("goalsRemaining", [])
    out.setdefault("recentMessages", [])
    out.setdefault("conversationTurn", 0)
    out.setdefault("shouldContinue", True)
    out.setdefault(
        "coachingStrategy",
        {"correctionMode": "SUBTLE", "followUpStyle": "OPEN"},
    )
    out.setdefault("userContext", {"profileFacts": [], "learningSignals": []})
    out.setdefault("pendingMemorySignals", [])
    out.setdefault("relevantMemories", [])
    out.setdefault("topicPlan", {})
    return out
