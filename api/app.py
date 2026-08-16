"""FastAPI — realtime English coach (Deepgram live + Groq + TTS over WS)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
import base64
import threading

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.call_ws import LiveCallBridge
from core import call_log
from core.coach_service import CoachService
from core.config import Settings, load_settings
from core.memory.vector import VectorMemory
from core.session import Session, new_session
from core.tfidf_engine import TfidfEngine
from core.topics import get_topic, list_topics
from wrappers.deepgram_tts import DeepgramTTS
from wrappers.llm import build_llm
from wrappers.mongo_store import MongoStore


class AppState:
    settings: Settings
    mongo: MongoStore
    coach: CoachService
    sessions: dict[str, Session]


STATE = AppState()


def _build_coach(settings: Settings, mongo: MongoStore) -> CoachService:
    llm = build_llm(settings)
    print(
        f"[api] llm provider={settings.llm_provider} "
        f"model={settings.sarvam_model if settings.llm_provider == 'sarvam' else settings.groq_model}"
    )
    return CoachService(
        tts=DeepgramTTS(settings.deepgram_tts_keys),
        coach=llm,
        tfidf=TfidfEngine(top_k=6),
        mongo=mongo,
        vectors=VectorMemory(mongo),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = load_settings()
    mongo = MongoStore(settings.mongo_uri, settings.mongo_db)
    STATE.settings = settings
    STATE.mongo = mongo
    STATE.coach = _build_coach(settings, mongo)
    STATE.sessions = {}

    def _seed() -> None:
        try:
            n = mongo.seed_topics()
            if n:
                print(f"[api] topics seeded once: {n}")
            else:
                print("[api] topics already seeded - skip")
        except Exception as exc:  # noqa: BLE001
            print(f"[api] topic seed skipped: {exc}")

    threading.Thread(target=_seed, daemon=True).start()
    print("[api] ready - realtime only (no local audio files)")
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
    topic_id: str | None = None
    learner_id: str = ""


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "realtime"}


@app.get("/api/topics")
def topics() -> dict[str, Any]:
    return {"topics": list_topics()}


@app.post("/api/sessions")
def start_session(body: StartSessionBody) -> dict[str, Any]:
    try:
        topic = get_topic(body.topic_id or "free-talk")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    session = new_session(
        mode="live",
        mongo=STATE.mongo,
        topic=topic,
        learner_id=body.learner_id,
    )
    opener = STATE.coach.open_session(session)
    STATE.sessions[session.session_id] = session
    call_log.info(
        "SESSION",
        f"started topic={topic.id}",
        session_id=session.session_id,
        extra={
            "topic": topic.id,
            "learner_id": session.learner_id,
            "tts_ms": (opener.latency or {}).get("tts_ms") if opener.latency else None,
        },
    )
    audio_b64 = (
        base64.b64encode(opener.coach_audio_bytes).decode("ascii")
        if opener.coach_audio_bytes
        else None
    )
    return {
        "session_id": session.session_id,
        "topic": topic.as_dict(),
        "turn": opener.turn,
        "coach_text": opener.coach_text,
        "coach_audio_b64": audio_b64,
        "keywords": opener.keywords,
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    doc = STATE.mongo.get_session(session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return doc


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
    )
    await bridge.run()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "ai-talk",
        "mode": "api",
        "health": "/api/health",
        "sessions": "/api/sessions",
        "call": "/ws/call/{session_id}",
    }
