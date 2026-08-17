"""FastAPI — realtime English coach (Deepgram live + Groq + TTS over WS)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.call_ws import LiveCallBridge
from core import call_log
from core.coach_service import CoachService
from core.config import Settings, load_settings
from core.session import Session, new_session
from core.tfidf_engine import TfidfEngine
from core.conversations.errors import ConversationError
from core.conversations.message_service import ConversationMessageService
from core.conversations.session_service import (
    ConversationSessionService,
    NORMAL_END_REASONS,
    REASON_USER_ENDED_CALL,
)
from core.conversations.summary_service import (
    ConversationSummaryService,
    run_summary_job,
)
from core.memory.learning_service import LearningMemoryService
from core.memory.profile_service import UserProfileMemoryService
from core.runtime.service import ConversationRuntimeService
from core.semantic.embeddings import build_embedding_provider
from core.semantic.service import SemanticMemoryService
from core.topics.errors import TopicProgressError
from core.topics.progress_service import TopicProgressService
from core.topics.view import public_practice_plan, public_topic
from wrappers.deepgram_tts import DeepgramTTS
from wrappers.llm import build_llm
from wrappers.mongo_store import MongoStore
from wrappers.qdrant_store import QdrantStore


class AppState:
    settings: Settings
    mongo: MongoStore | None
    qdrant: QdrantStore | None
    topic_progress: TopicProgressService | None
    conversation_sessions: ConversationSessionService | None
    conversation_messages: ConversationMessageService | None
    conversation_summaries: ConversationSummaryService | None
    profile_memory: UserProfileMemoryService | None
    learning_memory: LearningMemoryService | None
    semantic_memory: SemanticMemoryService | None
    conversation_runtime: ConversationRuntimeService | None
    coach: CoachService
    sessions: dict[str, Session]


STATE = AppState()


def _build_coach(settings: Settings) -> CoachService:
    llm = build_llm(settings)
    print(
        f"[api] llm provider={settings.llm_provider} "
        f"model={settings.sarvam_model if settings.llm_provider == 'sarvam' else settings.groq_model}"
    )
    return CoachService(
        tts=DeepgramTTS(settings.deepgram_tts_keys),
        coach=llm,
        tfidf=TfidfEngine(top_k=6),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = load_settings()
    STATE.settings = settings
    STATE.mongo = None
    STATE.qdrant = None
    STATE.topic_progress = None
    STATE.conversation_sessions = None
    STATE.conversation_messages = None
    STATE.conversation_summaries = None
    STATE.profile_memory = None
    STATE.learning_memory = None
    STATE.semantic_memory = None
    STATE.conversation_runtime = None
    if settings.mongo_uri:
        try:
            mongo = MongoStore(
                settings.mongo_uri,
                settings.mongo_db,
                users_db=settings.users_mongo_db,
            )
            mongo.ping()
            mongo.ensure_schema()
            STATE.mongo = mongo
            STATE.topic_progress = TopicProgressService(mongo)
            STATE.conversation_sessions = ConversationSessionService(mongo)
            STATE.conversation_messages = ConversationMessageService(mongo)
        except Exception as exc:  # noqa: BLE001
            print(f"[api] mongo schema skip: {exc}")
            call_log.warn("MONGO", f"schema skip: {exc}")
    else:
        print("[api] MONGODB_URI empty — schema not applied")
    qdrant = QdrantStore(
        settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        vector_size=settings.embedding_dimension,
        distance=settings.qdrant_distance,
    )
    STATE.qdrant = qdrant
    try:
        qdrant.ensure_collection()
    except Exception as exc:  # noqa: BLE001
        print(f"[api] qdrant schema skip: {exc}")
        call_log.warn("VECTOR", f"schema skip: {exc}")
    STATE.coach = _build_coach(settings)
    if STATE.mongo is not None:
        STATE.conversation_summaries = ConversationSummaryService(
            STATE.mongo,
            analyzer=STATE.coach.coach,
        )
        STATE.profile_memory = UserProfileMemoryService(
            STATE.mongo,
            analyzer=STATE.coach.coach,
        )
        STATE.learning_memory = LearningMemoryService(
            STATE.mongo,
            analyzer=STATE.coach.coach,
        )
        STATE.semantic_memory = SemanticMemoryService(
            STATE.mongo,
            vectors=STATE.qdrant,
            embeddings=build_embedding_provider(
                provider=settings.embedding_provider,
                model=settings.embedding_model,
                dimension=settings.embedding_dimension,
                version=settings.embedding_version,
            ),
            analyzer=STATE.coach.coach,
            tenant_id=settings.tenant_id,
            embedding_model=settings.embedding_model,
            embedding_version=settings.embedding_version,
        )
        STATE.conversation_runtime = ConversationRuntimeService(
            STATE.mongo,
            analyzer=STATE.coach.coach,
            semantic=STATE.semantic_memory,
            tenant_id=settings.tenant_id,
        )
    STATE.sessions = {}
    print("[api] ready - realtime coach (topics + memory on live call)")
    call_log.info("SERVER", f"ready log={call_log.log_path()}")
    yield


app = FastAPI(title="AI Talk English Coach", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartSessionBody(BaseModel):
    learner_id: str
    topic_id: str | None = None


def _load_practice_plan(learner_id: str) -> dict[str, Any]:
    svc = STATE.topic_progress
    if svc is None:
        raise HTTPException(status_code=503, detail="TOPIC_SERVICE_UNAVAILABLE")
    try:
        return svc.getPracticePlan(learner_id)
    except TopicProgressError as exc:
        status = 404 if exc.code in {"USER_NOT_FOUND", "TOPICS_NOT_FOUND_FOR_LEVEL"} else 400
        raise HTTPException(status_code=status, detail=exc.code) from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "realtime"}


@app.get("/api/topics")
def topics(level: str | None = Query(default=None)) -> dict[str, Any]:
    mongo = STATE.mongo
    if mongo is None or not hasattr(mongo, "list_curriculum_topics"):
        return {"topics": []}
    rows = mongo.list_curriculum_topics(level=level)
    return {"topics": [public_topic(row) for row in rows if public_topic(row)]}


@app.get("/api/practice-plan")
def practice_plan(learner_id: str = Query(...)) -> dict[str, Any]:
    user_id = (learner_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="USER_NOT_FOUND")
    plan = _load_practice_plan(user_id)
    return public_practice_plan(plan) or {"topic": None}


@app.post("/api/sessions")
def start_session(body: StartSessionBody) -> dict[str, Any]:
    """In-memory live socket key. Topic Engine chooses the topic. No generic opener."""
    learner_id = (body.learner_id or "").strip()
    if not learner_id:
        raise HTTPException(status_code=400, detail="USER_NOT_FOUND")
    mongo = STATE.mongo
    if mongo is not None and hasattr(mongo, "find_user") and mongo.find_user(learner_id) is None:
        raise HTTPException(status_code=404, detail="USER_NOT_FOUND")
    session = new_session(mode="live", learner_id=learner_id)
    STATE.sessions[session.session_id] = session
    plan = None
    if STATE.topic_progress is not None:
        plan = public_practice_plan(_load_practice_plan(learner_id))
    call_log.info(
        "SESSION",
        "started",
        session_id=session.session_id,
        extra={"learner_id": session.learner_id},
    )
    return {
        "session_id": session.session_id,
        "learner_id": session.learner_id,
        "ws_url": f"/ws/call/{session.session_id}",
        "topic": (plan or {}).get("topic"),
        "practicePlan": plan,
        "turn": 0,
        "coach_text": None,
        "coach_audio_b64": None,
        "keywords": [],
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = STATE.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.session_id,
        "mode": session.mode,
        "learner_id": session.learner_id,
        "turn": session.turn,
        "messages": session.messages,
        "keywords": session.keywords,
        "conversation_id": session.conversation_id,
    }


@app.websocket("/ws/call/{session_id}")
async def call_socket(websocket: WebSocket, session_id: str) -> None:
    session = STATE.sessions.get(session_id)
    if session is None:
        await websocket.accept()
        await websocket.send_json(
            {"type": "error", "detail": "Session not found. Start a session again."}
        )
        await websocket.close(code=4404)
        call_log.error("CONNECT", "session not found", session_id=session_id)
        return
    bridge = LiveCallBridge(
        websocket=websocket,
        session=session,
        coach=STATE.coach,
        settings=STATE.settings,
        topic_progress=STATE.topic_progress,
        conversation_sessions=STATE.conversation_sessions,
        conversation_messages=STATE.conversation_messages,
        conversation_summaries=STATE.conversation_summaries,
        profile_memory=STATE.profile_memory,
        learning_memory=STATE.learning_memory,
        conversation_runtime=STATE.conversation_runtime,
        semantic_memory=STATE.semantic_memory,
    )
    await bridge.run()


class EndConversationBody(BaseModel):
    learner_id: str
    reason: str | None = None


@app.post("/api/conversations/{conversation_id}/end")
def end_conversation(
    conversation_id: str,
    body: EndConversationBody,
    background: BackgroundTasks,
) -> dict[str, Any]:
    """Fallback if the live socket already dropped. WS disconnect is the source of truth."""
    svc = STATE.conversation_sessions
    if svc is None:
        raise HTTPException(status_code=503, detail="CONVERSATION_SERVICE_UNAVAILABLE")
    user_id = (body.learner_id or "").strip()
    if not user_id:
        raise HTTPException(status_code=403, detail="CONVERSATION_ACCESS_DENIED")
    reason = (body.reason or "").strip() or REASON_USER_ENDED_CALL
    try:
        if reason in NORMAL_END_REASONS:
            result = svc.completeConversationSession(
                conversation_id,
                {"reason": reason},
                userId=user_id,
            )
        else:
            result = svc.failConversationSession(
                conversation_id,
                reason,
                userId=user_id,
            )
    except ConversationError as exc:
        status_code = 404 if exc.code == "CONVERSATION_NOT_FOUND" else 403
        if exc.code not in {"CONVERSATION_NOT_FOUND", "CONVERSATION_ACCESS_DENIED"}:
            status_code = 400
        raise HTTPException(status_code=status_code, detail=exc.code) from exc
    if result.get("status") == "COMPLETED":
        background.add_task(
            run_summary_job,
            STATE.conversation_summaries,
            conversation_id,
            user_id=user_id,
            progress_service=STATE.topic_progress,
            profile_service=STATE.profile_memory,
            learning_service=STATE.learning_memory,
            semantic_service=STATE.semantic_memory,
        )
    return {
        "conversationId": result.get("conversationId"),
        "status": result.get("status"),
        "startedAt": result.get("startedAt"),
        "endedAt": result.get("endedAt"),
        "durationSeconds": result.get("durationSeconds"),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "ai-talk",
        "mode": "api",
        "health": "/api/health",
        "sessions": "/api/sessions",
        "call": "/ws/call/{session_id}",
    }
