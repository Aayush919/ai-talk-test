"""Realtime call — looping pipeline: STT partial → LLM → TTS buffer → flush."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.listen.v1.types.listen_v1speech_started import ListenV1SpeechStarted
from deepgram.listen.v1.types.listen_v1utterance_end import ListenV1UtteranceEnd
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from core.coach_service import (
    CoachService,
    DraftReply,
    needs_fresh_reply,
    transcripts_compatible,
)
from core.conversation.engagement import is_low_content_turn
from core.conversation.stt_merge import merge_pending_stt
from core import call_log
from core.config import Settings
from core.session import Session
from core.text_clean import clip_spoken_reply, safe_print
from core.conversations.message_service import ConversationMessageService
from core.conversations.session_service import (
    CALL_TYPE_AI_COACH,
    ConversationSessionService,
    REASON_NETWORK_FAILURE,
    REASON_TIMEOUT,
    REASON_USER_ENDED_CALL,
    topic_id_of,
)
from core.topics.view import public_practice_plan
from core.conversations.summary_service import (
    ConversationSummaryService,
    run_summary_job,
)
from core.runtime.graph import FALLBACK_RESPONSE
from core.topics.errors import TopicProgressError
from core.topics.progress_service import TopicProgressService
from wrappers.deepgram_tts import split_speak_chunks

OPENING_FALLBACK = (
    "Hi. I'm your English speaking coach. Tell me a little about yourself."
)

# local import of _norm via needs / compatible already enough
from core.coach_service import _norm


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    try:
        from bson import ObjectId

        if isinstance(value, ObjectId):
            return str(value)
    except Exception:
        pass
    return value

print = safe_print  # Windows cp1252-safe (no charmap crash on fancy dashes)


@dataclass
class SpecPipeline:
    """Warm LLM reply + Deepgram PCM while user is still speaking."""

    source: str
    token: int
    user_text: str = ""
    coach_text: str = ""
    keywords: list[str] = field(default_factory=list)
    llm_ms: int = 0
    units: list[str] = field(default_factory=list)
    units_done: int = 0
    pcm: bytearray = field(default_factory=bytearray)
    audio_ready: bool = False
    done: bool = False


@dataclass
class SpecSnap:
    user_text: str
    coach_text: str
    keywords: list[str]
    llm_ms: int
    units: list[str]
    units_done: int
    pcm: bytes
    done: bool


def _transcript_from_results(message: ListenV1Results) -> str:
    channel = message.channel
    alts = getattr(channel, "alternatives", None) or []
    if not alts:
        return ""
    return (getattr(alts[0], "transcript", None) or "").strip()


def _confidence_from_results(message: ListenV1Results) -> float | None:
    channel = message.channel
    alts = getattr(channel, "alternatives", None) or []
    if not alts:
        return None
    value = getattr(alts[0], "confidence", None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _peak_16le(chunk: bytes, bridge: Any) -> None:
    """Cheap loudness probe — tells a muted mic apart from a missing mic."""
    if bridge._audio_bytes > 400_000:  # ~12s of audio is plenty to judge
        return
    peak = bridge._audio_peak
    for i in range(0, len(chunk) - 1, 64):
        sample = int.from_bytes(chunk[i : i + 2], "little", signed=True)
        if sample < 0:
            sample = -sample
        if sample > peak:
            peak = sample
    bridge._audio_peak = peak


def _same_utterance(prev: str, nxt: str) -> bool:
    """True if nxt continues prev — not a second thought glued onto pending STT."""
    a, b = _norm(prev), _norm(nxt)
    if not a or not b:
        return True
    if b.startswith(a) or a.startswith(b):
        return True
    overlap = set(a.split()[-5:]) & set(b.split()[:6])
    return len(overlap) >= 2


class LiveCallBridge:
    """
    Looping pipeline while user speaks:
      STT partial → Groq draft → Deepgram PCM buffer
    On speech_final: flush buffered audio (pipeline_hit) or cold stream.
    """

    def __init__(
        self,
        *,
        websocket: WebSocket,
        session: Session,
        coach: CoachService,
        settings: Settings,
        topic_progress: TopicProgressService | None = None,
        conversation_sessions: ConversationSessionService | None = None,
        conversation_messages: ConversationMessageService | None = None,
        conversation_summaries: ConversationSummaryService | None = None,
        profile_memory=None,
        learning_memory=None,
        conversation_runtime=None,
        semantic_memory=None,
    ) -> None:
        self.ws = websocket
        self.session = session
        self.coach = coach
        self.settings = settings
        self.topic_progress = topic_progress
        self.conversation_sessions = conversation_sessions
        self.conversation_messages = conversation_messages
        self.conversation_summaries = conversation_summaries
        self.profile_memory = profile_memory
        self.learning_memory = learning_memory
        self.conversation_runtime = conversation_runtime
        self.semantic_memory = semantic_memory
        self._last_target_goal_id: str | None = None
        self._last_stt_confidence: float | None = None
        self._audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=128)
        self._pending = ""
        self._queued = ""
        self._turn_task: asyncio.Task[None] | None = None
        self._spec_task: asyncio.Task[None] | None = None
        self._spec: SpecPipeline | None = None
        self._spec_token = 0
        self._generation = 0
        self._closed = False
        self._speaking = False
        self._speak_started_at = 0.0
        self._speak_until = 0.0
        self._hold_task: asyncio.Task[None] | None = None
        self._dg = None
        self._last_partial = ""
        self._spec_debounce: asyncio.Task[None] | None = None
        # Soft barge-in: ignore echo for a moment, then allow real interrupt
        self._barge_grace_s = 0.85
        self._barge_min_words = 2
        self._barge_task: asyncio.Task[None] | None = None
        self._last_event = "init"
        self._turn_started_at = 0.0
        # Silence lifecycle: quiet 10s -> warn, quiet 30s more -> hang up
        self._idle_warn_s = 10.0
        self._idle_end_s = 30.0
        self._silence_at = time.perf_counter()
        self._warned = False
        self._audio_bytes = 0
        self._audio_peak = 0
        self._mic_flagged = False
        self._hangup_ok = False
        self._end_reason = REASON_NETWORK_FAILURE
        self._persist_tasks: set[asyncio.Task[Any]] = set()
        self._persist_chain: asyncio.Task[Any] | None = None

    def _sid(self) -> str:
        return getattr(self.session, "session_id", "") or ""

    def _dbg(self, msg: str) -> None:
        """Console + logs/ai-talk.log + logs/call-<sid>.log"""
        self._last_event = msg
        upper = msg.upper()
        level = "error" if (
            any(w in upper for w in ("FAIL", "STUCK", "WARN"))
            or ("ERROR" in upper and "CANCELLEDERROR" not in upper)
        ) else "info"
        writer = call_log.error if level == "error" else call_log.info
        writer("CALL", msg, session_id=self._sid())
        print(f"[calldbg] {msg} | {self._state_line()}")

    def _log_lat(self, latency: dict[str, Any]) -> None:
        call_log.lat(
            "LATENCY",
            f"hear@{latency.get('wait_ms', latency.get('total_ms', '?'))}ms",
            session_id=self._sid(),
            extra={
                "mode": latency.get("mode"),
                "llm_ms": latency.get("llm_ms"),
                "llm_ttfb_ms": latency.get("llm_ttfb_ms", latency.get("llm_ms")),
                "ttfb_ms": latency.get("ttfb_ms"),
                "tts_ms": latency.get("tts_ms"),
                "flush_ms": latency.get("flush_ms"),
                "spec": latency.get("speculative"),
                "total_turn_ms": latency.get("total_ms") or latency.get("wait_ms"),
            },
        )

    def _state_line(self) -> str:
        spec = self._spec
        spec_bits = "none"
        if spec:
            spec_bits = (
                f"src={spec.source[:40]!r} llm={bool(spec.coach_text)} "
                f"audio={spec.audio_ready} units={spec.units_done}/{len(spec.units)} "
                f"bytes={len(spec.pcm)} done={spec.done}"
            )
        turn_age = (
            int((time.perf_counter() - self._turn_started_at) * 1000)
            if self._turn_started_at
            else 0
        )
        return (
            f"speak={self._speaking} gen={self._generation} "
            f"turn_busy={bool(self._turn_task and not self._turn_task.done())} "
            f"turn_age={turn_age}ms "
            f"spec_busy={bool(self._spec_task and not self._spec_task.done())} "
            f"pending={self._pending[:40]!r} queued={self._queued[:40]!r} "
            f"partial={self._last_partial[:40]!r} spec=[{spec_bits}]"
        )

    async def _heartbeat(self) -> None:
        """Every 3s dump state — catches silent hangs during a call."""
        while not self._closed:
            await asyncio.sleep(3.0)
            if self._closed:
                return
            stuck = ""
            if self._turn_task and not self._turn_task.done() and self._turn_started_at:
                age = time.perf_counter() - self._turn_started_at
                if age > 6.0:
                    stuck = f" STUCK_TURN? age={age:.1f}s"
                    await self.send_json(
                        {
                            "type": "diag",
                            "detail": f"turn still running {age:.1f}s — {self._last_event}",
                        }
                    )
            if self._speaking and self._speak_started_at:
                sage = time.perf_counter() - self._speak_started_at
                if sage > 20.0:
                    stuck += f" STUCK_SPEAK? age={sage:.1f}s"
            if stuck:
                call_log.warn("HEARTBEAT", stuck.strip(), session_id=self._sid())
            print(f"[calldbg] HEARTBEAT{stuck} | {self._state_line()}")

    def _touch_voice(self) -> None:
        self._silence_at = time.perf_counter()
        self._warned = False

    async def _idle_watch(self) -> None:
        """Learner quiet 10s -> one warning; quiet 30s after that -> end the call."""
        while not self._closed:
            await asyncio.sleep(0.5)
            if self._closed:
                return
            busy = self._coach_busy() or bool(
                self._turn_task and not self._turn_task.done()
            )
            if busy:
                # Coach audio / thinking is not learner silence — restart the clock
                self._silence_at = time.perf_counter()
                continue
            quiet = time.perf_counter() - self._silence_at
            if not self._mic_flagged and quiet >= 5.0:
                self._mic_flagged = True
                if self._audio_bytes == 0:
                    call_log.warn(
                        "MIC",
                        "no audio bytes from browser in first 5s",
                        session_id=self._sid(),
                    )
                    await self.send_json(
                        {"type": "info", "detail": "No mic audio reaching the server."}
                    )
                elif self._audio_peak < 300:
                    call_log.warn(
                        "MIC",
                        "audio arriving but nearly silent — check input device",
                        session_id=self._sid(),
                        extra={
                            "bytes": self._audio_bytes,
                            "peak": self._audio_peak,
                        },
                    )
                    await self.send_json(
                        {
                            "type": "info",
                            "detail": "Mic audio is silent — wrong input device?",
                        }
                    )
                else:
                    call_log.info(
                        "MIC",
                        "audio ok but no transcript yet",
                        session_id=self._sid(),
                        extra={
                            "bytes": self._audio_bytes,
                            "peak": self._audio_peak,
                        },
                    )
            limit = self._idle_end_s if self._warned else self._idle_warn_s
            if quiet < limit:
                continue
            if not self._warned:
                self._warned = True
                self._dbg(
                    f"silence warn after {quiet:.1f}s "
                    f"(mic bytes={self._audio_bytes} peak={self._audio_peak})"
                )
                await self._speak_line(
                    "Are you still there? I'll wait thirty more seconds, "
                    "otherwise I'll end the call."
                )
                self._silence_at = time.perf_counter()
            else:
                self._dbg(f"silence end after {quiet:.1f}s")
                await self._end_for_silence()
                return

    async def _speak_line(self, line: str) -> None:
        """Speak a system line (silence warning) on the live stream protocol."""
        self._generation += 1
        generation = self._generation
        self._speaking = True
        self._speak_started_at = time.perf_counter()
        try:
            await self.send_json(
                {
                    "type": "coach_turn",
                    "turn": self.session.turn,
                    "user_text": "",
                    "coach_text": line,
                    "keywords": [],
                    "stream": True,
                    "audio_format": "pcm_s16le",
                    "sample_rate": 16000,
                    "mode": "system",
                }
            )
            async for chunk in self.coach.tts.stream_pcm_chunked(line):
                if self._closed or generation != self._generation:
                    return
                try:
                    await self.ws.send_bytes(chunk)
                except Exception:
                    break
            await self.send_json({"type": "coach_audio_end", "latency": {"mode": "system"}})
        except Exception as exc:  # noqa: BLE001
            self._dbg(f"system line failed: {exc}")
        finally:
            if generation == self._generation:
                self._hold_playback(line)
            else:
                self._speaking = False

    async def _end_for_silence(self) -> None:
        self._hangup_ok = True
        self._end_reason = REASON_TIMEOUT
        await self._speak_line("No answer, so I'm ending the call. Talk to you soon!")
        await asyncio.sleep(max(0.0, self._speak_until - time.perf_counter()))
        call_log.info(
            "DISCONNECT",
            "ended on silence",
            session_id=self._sid(),
            extra={"warn_s": self._idle_warn_s, "end_s": self._idle_end_s},
        )
        await self.send_json({"type": "call_ended", "reason": "silence"})
        self._closed = True
        try:
            await self._audio_q.put(None)
        except Exception:
            pass
        try:
            if self.ws.client_state == WebSocketState.CONNECTED:
                await self.ws.close(code=1000)
        except Exception:
            pass

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.ws.client_state != WebSocketState.CONNECTED or self._closed:
            return
        try:
            await self.ws.send_json(payload)
        except Exception:
            pass

    async def _init_current_topic(self) -> dict[str, Any] | None:
        """After the live socket is up — never on register / session create."""
        if self.topic_progress is None:
            return None
        user_id = (self.session.learner_id or "").strip()
        if not user_id:
            await self.send_json(
                {"type": "error", "code": "USER_NOT_FOUND", "detail": "USER_NOT_FOUND"}
            )
            return None
        try:
            result = await asyncio.to_thread(
                getattr(
                    self.topic_progress,
                    "getPracticePlan",
                    self.topic_progress.getOrInitializeCurrentTopic,
                ),
                user_id,
            )
            self.session.current_topic = result
            call_log.info(
                "TOPIC",
                "current topic ready",
                session_id=self._sid(),
                extra={
                    "initialized": result.get("initialized"),
                    "slug": (result.get("topic") or {}).get("slug"),
                },
            )
            return _json_safe(result)
        except TopicProgressError as exc:
            call_log.warn("TOPIC", exc.code, session_id=self._sid())
            await self.send_json(
                {"type": "error", "code": exc.code, "detail": exc.code}
            )
            return None
        except Exception as exc:  # noqa: BLE001
            call_log.warn("TOPIC", f"init skip: {exc}", session_id=self._sid())
            await self.send_json(
                {
                    "type": "error",
                    "code": "TOPIC_PROGRESS_INTERNAL_ERROR",
                    "detail": "TOPIC_PROGRESS_INTERNAL_ERROR",
                }
            )
            return None

    async def _create_conversation_session(self) -> dict[str, Any] | None:
        """After a successful socket + current topic — never on register / invite."""
        svc = self.conversation_sessions
        if svc is None:
            return None
        user_id = (self.session.learner_id or "").strip()
        topic_id = topic_id_of(self.session.current_topic)
        if not user_id or topic_id is None:
            return None
        try:
            result = await asyncio.to_thread(
                lambda: svc.createConversationSession(
                    userId=user_id,
                    topicId=topic_id,
                    callType=CALL_TYPE_AI_COACH,
                )
            )
            self.session.conversation_id = result["conversationId"]
            if self.conversation_runtime is not None:
                try:
                    await asyncio.to_thread(
                        self.conversation_runtime.initializeConversationRuntime,
                        result["conversationId"],
                    )
                except Exception as exc:  # noqa: BLE001
                    call_log.warn(
                        "RUNTIME",
                        f"init skip: {exc}",
                        session_id=self._sid(),
                    )
            call_log.info(
                "SESSION",
                "conversation session active",
                session_id=self._sid(),
                extra={"conversationId": result["conversationId"]},
            )
            return _json_safe(result)
        except TopicProgressError as exc:
            call_log.warn("SESSION", exc.code, session_id=self._sid())
            await self.send_json(
                {"type": "error", "code": exc.code, "detail": exc.code}
            )
            return None
        except Exception as exc:  # noqa: BLE001
            call_log.warn("SESSION", f"create skip: {exc}", session_id=self._sid())
            await self.send_json(
                {
                    "type": "error",
                    "code": "TOPIC_PROGRESS_INTERNAL_ERROR",
                    "detail": "TOPIC_PROGRESS_INTERNAL_ERROR",
                }
            )
            return None

    def _draft_reply(self, text: str) -> DraftReply:
        t0 = time.perf_counter()
        cid = self.session.conversation_id
        runtime = self.conversation_runtime
        if runtime is not None and cid:
            try:
                decision = runtime.previewResponse(
                    cid, text, sttConfidence=self._last_stt_confidence
                )
                self._last_target_goal_id = decision.get("targetGoalId")
                reply = clip_spoken_reply(
                    str(decision.get("response") or decision.get("text") or "").strip(),
                    user_text=text,
                )
                llm_ms = int(decision.get("llm_ms") or 0)
                if llm_ms <= 0:
                    llm_ms = int((time.perf_counter() - t0) * 1000)
                if reply:
                    history_texts = [m["content"] for m in self.session.messages]
                    keywords = self.coach.tfidf.extract(history_texts, text)
                    return DraftReply(
                        user_text=text,
                        coach_text=reply,
                        keywords=keywords,
                        llm_ms=llm_ms,
                    )
            except Exception as exc:  # noqa: BLE001
                call_log.warn(
                    "RUNTIME",
                    f"preview skip: {type(exc).__name__}: {exc}",
                    session_id=self._sid(),
                    extra={"latency_ms": int((time.perf_counter() - t0) * 1000)},
                )
            return DraftReply(
                user_text=text,
                coach_text=FALLBACK_RESPONSE,
                keywords=[],
                llm_ms=int((time.perf_counter() - t0) * 1000),
            )
        draft = self.coach.draft_reply(self.session, text)
        if draft.llm_ms <= 0:
            draft.llm_ms = int((time.perf_counter() - t0) * 1000)
        return draft

    def _commit_turn(
        self,
        *,
        user_text: str,
        coach_text: str,
        keywords: list[str],
    ) -> int:
        turn = self.coach.commit_text(
            self.session,
            user_text=user_text,
            coach_text=coach_text,
            keywords=keywords,
        )
        cid = self.session.conversation_id
        runtime = self.conversation_runtime
        if runtime is not None and cid:
            try:
                runtime.applyCommittedTurn(
                    cid,
                    userText=user_text,
                    assistantText=coach_text,
                    targetGoalId=self._last_target_goal_id,
                )
            except Exception as exc:  # noqa: BLE001
                call_log.warn("RUNTIME", f"commit skip: {exc}", session_id=self._sid())
        return turn

    async def _finalize_conversation_session(self) -> None:
        """Single backend cleanup: normal hangup → COMPLETED, anything else → FAILED."""
        cid = self.session.conversation_id
        svc = self.conversation_sessions
        if not cid or svc is None:
            return
        runtime = self.conversation_runtime
        runtime_snapshot = None
        if runtime is not None:
            try:
                runtime_snapshot = await asyncio.to_thread(runtime.getRuntimeState, cid)
            except Exception as exc:  # noqa: BLE001
                call_log.warn("RUNTIME", f"snapshot skip: {exc}", session_id=self._sid())
            try:
                await asyncio.to_thread(runtime.endConversationRuntime, cid)
            except Exception as exc:  # noqa: BLE001
                call_log.warn("RUNTIME", f"end skip: {exc}", session_id=self._sid())
        user_id = (self.session.learner_id or "").strip() or None
        reason = self._end_reason
        try:
            if self._hangup_ok:
                result = await asyncio.to_thread(
                    lambda: svc.completeConversationSession(
                        cid,
                        {"reason": reason},
                        userId=user_id,
                    )
                )
            else:
                result = await asyncio.to_thread(
                    lambda: svc.failConversationSession(
                        cid,
                        reason,
                        userId=user_id,
                    )
                )
            call_log.info(
                "SESSION",
                f"conversation session {result.get('status')}",
                session_id=self._sid(),
                extra={
                    "conversationId": cid,
                    "userId": result.get("userId") or user_id,
                    "topicId": result.get("topicId"),
                    "status": result.get("status"),
                    "durationSeconds": result.get("durationSeconds"),
                    "reason": result.get("reason") or reason,
                },
            )
            if result.get("status") == "COMPLETED":
                asyncio.create_task(
                    asyncio.to_thread(
                        run_summary_job,
                        self.conversation_summaries,
                        cid,
                        user_id=user_id,
                        progress_service=self.topic_progress,
                        profile_service=self.profile_memory,
                        learning_service=self.learning_memory,
                        semantic_service=self.semantic_memory,
                        runtime_snapshot=runtime_snapshot,
                    ),
                    name="conversation-summary",
                )
        except Exception as exc:  # noqa: BLE001
            call_log.warn("SESSION", f"finalize skip: {exc}", session_id=self._sid())

    def _persist_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Queue Mongo write in call order; never block STT/LLM/TTS."""
        cid = self.session.conversation_id
        svc = self.conversation_messages
        text = (content or "").strip()
        if not cid or svc is None or not text:
            return
        user_id = (self.session.learner_id or "").strip() or None
        previous = self._persist_chain

        async def _job() -> None:
            if previous is not None:
                try:
                    await previous
                except Exception:
                    pass
            try:
                await asyncio.to_thread(
                    lambda: svc.createConversationMessage(
                        conversationId=cid,
                        role=role,
                        content=text,
                        metadata=metadata,
                        userId=user_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                cause = getattr(exc, "__cause__", None)
                call_log.error(
                    "MESSAGE",
                    f"persist failed: {type(exc).__name__}: {exc}",
                    session_id=self._sid(),
                    extra={
                        "conversationId": cid,
                        "userId": user_id,
                        "role": role,
                        "cause": f"{type(cause).__name__}: {cause}" if cause else "",
                    },
                )

        task = asyncio.create_task(_job(), name="persist-message")
        self._persist_chain = task
        self._persist_tasks.add(task)
        task.add_done_callback(self._persist_tasks.discard)

    def _fallback_opening(self) -> str:
        return OPENING_FALLBACK

    async def _speak_opening(self) -> None:
        """AI talks first after connect. Never wait for the user to speak."""
        cid = self.session.conversation_id
        runtime = self.conversation_runtime
        text = ""
        llm_ms = 0
        if runtime is not None and cid:
            try:
                opening = await asyncio.to_thread(runtime.generateOpening, cid)
                llm_ms = int((opening or {}).get("llm_ms") or 0)
                text = clip_spoken_reply(str((opening or {}).get("text") or "").strip())
                call_log.info(
                    "LATENCY",
                    "OPENING_LLM_COMPLETE",
                    session_id=self._sid(),
                    extra={"llm_ms": llm_ms, "llm_ttfb_ms": llm_ms},
                )
            except Exception as exc:  # noqa: BLE001
                call_log.warn("RUNTIME", f"opening llm skip: {exc}", session_id=self._sid())
        else:
            call_log.warn(
                "RUNTIME",
                "opening using fallback — runtime or conversation missing",
                session_id=self._sid(),
            )
        if not text or text == FALLBACK_RESPONSE:
            text = self._fallback_opening()
            if runtime is not None and cid:
                try:
                    runtime.graph.app.update_state(
                        {"configurable": {"thread_id": cid}},
                        {
                            "lastAssistantMessage": text,
                            "lastAssistantQuestion": text if "?" in text else None,
                            "conversationPhase": "WARMUP",
                        },
                    )
                except Exception:
                    pass
        self._dbg(f"opening speak {text!r}")
        self.session.add("assistant", text)
        self._persist_message(
            "assistant",
            text,
            {"source": "ai", "ttsProvider": "deepgram", "kind": "opening"},
        )
        try:
            await self.send_json(
                {
                    "type": "coach_turn",
                    "turn": 0,
                    "user_text": "",
                    "coach_text": text,
                    "keywords": [],
                    "stream": True,
                    "audio_format": "pcm_s16le",
                    "sample_rate": 16000,
                    "mode": "opening",
                }
            )
            self._speaking = True
            self._speak_started_at = time.perf_counter()
            async for chunk in self.coach.tts.stream_pcm_chunked(text):
                if self._closed:
                    break
                try:
                    await self.ws.send_bytes(chunk)
                except Exception:
                    break
            await self.send_json({"type": "coach_audio_end", "mode": "opening"})
        except Exception as exc:  # noqa: BLE001
            call_log.warn("RUNTIME", f"opening tts skip: {exc}", session_id=self._sid())
        finally:
            self._hold_playback(text)

    async def _drain_persist_tasks(self) -> None:
        chain = self._persist_chain
        if chain is not None and not chain.done():
            try:
                await asyncio.wait_for(asyncio.shield(chain), timeout=10.0)
            except asyncio.TimeoutError:
                call_log.warn(
                    "MESSAGE",
                    "persist still running at hangup",
                    session_id=self._sid(),
                )
            except Exception:
                pass
        pending = [task for task in self._persist_tasks if not task.done()]
        if not pending:
            return
        await asyncio.wait(pending, timeout=2.0)

    async def run(self) -> None:
        await self.ws.accept()
        topic_payload = await self._init_current_topic()
        conversation = None
        if topic_payload:
            conversation = await self._create_conversation_session()
        ready = {
            "type": "call_ready",
            "session_id": self.session.session_id,
            "message": "Live call connected — speak anytime.",
        }
        plan = public_practice_plan(topic_payload) if topic_payload else None
        if plan:
            ready["topic"] = plan.get("topic")
            ready["practicePlan"] = plan
            ready["initialized"] = plan.get("initialized")
        if conversation:
            ready["conversationId"] = conversation.get("conversationId")
            ready["conversation"] = conversation
        await self.send_json(ready)
        await self._speak_opening()

        recv_task = asyncio.create_task(self._recv_browser(), name="recv")
        dg_task = asyncio.create_task(self._deepgram_loop(), name="deepgram")
        hb_task = asyncio.create_task(self._heartbeat(), name="heartbeat")
        idle_task = asyncio.create_task(self._idle_watch(), name="idle")
        call_log.info(
            "CONNECT",
            "live call connected",
            session_id=self._sid(),
            extra={
                "logfile": str(call_log.session_log_path(self._sid())),
            },
        )
        self._dbg("call connected")

        try:
            await recv_task
        except WebSocketDisconnect:
            self._dbg("browser disconnect")
        except Exception as exc:  # noqa: BLE001
            call_log.error("WS", f"recv error: {exc}", session_id=self._sid())
            self._dbg(f"recv error: {exc}")
            await self.send_json({"type": "error", "detail": str(exc)})
        finally:
            self._closed = True
            call_log.info("DISCONNECT", "call ended", session_id=self._sid())
            await self._drain_persist_tasks()
            await self._finalize_conversation_session()
            await self._audio_q.put(None)
            self._cancel_spec()
            if self._turn_task and not self._turn_task.done():
                self._turn_task.cancel()
            hb_task.cancel()
            dg_task.cancel()
            idle_task.cancel()
            await asyncio.gather(dg_task, hb_task, idle_task, return_exceptions=True)
            try:
                await self.coach.tts.aclose()
            except Exception:
                pass
            try:
                if self.ws.client_state == WebSocketState.CONNECTED:
                    await self.ws.close()
            except Exception:
                pass

    async def _recv_browser(self) -> None:
        while not self._closed:
            try:
                message = await self.ws.receive()
            except WebSocketDisconnect:
                break
            except Exception as exc:  # noqa: BLE001
                call_log.warn("WS", f"receive skip: {exc}", session_id=self._sid())
                continue
            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data:
                self._audio_bytes += len(data)
                _peak_16le(data, self)
                try:
                    self._audio_q.put_nowait(data)
                except asyncio.QueueFull:
                    try:
                        _ = self._audio_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        self._audio_q.put_nowait(data)
                    except asyncio.QueueFull:
                        pass
                continue

            text = message.get("text")
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            try:
                if payload.get("type") == "end_call":
                    self._hangup_ok = True
                    self._end_reason = REASON_USER_ENDED_CALL
                    break
                if payload.get("type") == "ping":
                    await self.send_json({"type": "pong"})
                    continue
                if payload.get("type") == "client_log":
                    lvl = str(payload.get("level") or "info").lower()
                    writer = call_log.error if lvl == "error" else (
                        call_log.warn if lvl == "warn" else call_log.info
                    )
                    raw_fields = (
                        payload.get("fields")
                        if isinstance(payload.get("fields"), dict)
                        else {}
                    )
                    writer(
                        "BROWSER",
                        str(payload.get("detail") or payload.get("name") or ""),
                        session_id=self._sid(),
                        extra={
                            "name": payload.get("event") or payload.get("name"),
                            **{
                                k: v
                                for k, v in raw_fields.items()
                                if k in (
                                    "hear_ms",
                                    "remain_ms",
                                    "mode",
                                    "wait_ms",
                                    "llm_ms",
                                    "ttfb_ms",
                                    "tts_ms",
                                )
                            },
                        },
                    )
                    continue
            except Exception as exc:  # noqa: BLE001
                call_log.warn("WS", f"browser msg skip: {exc}", session_id=self._sid())
                continue

    async def _deepgram_loop(self) -> None:
        chain = list(self.settings.deepgram_stt.language_chain) or ["en"]
        languages: list[str] = []
        for code in chain:
            mapped = "en" if code.lower().startswith("en-in") else code
            if mapped not in languages:
                languages.append(mapped)
        for extra in ("en", "en-US"):
            if extra not in languages:
                languages.append(extra)

        endpointing = min(int(self.settings.deepgram_stt.endpointing_ms), 200)

        while not self._closed:
            connected = False
            for language in languages:
                if self._closed:
                    return
                try:
                    await self._run_deepgram_session(language, endpointing)
                    connected = True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    detail = str(exc)
                    if "status_code: 400" in detail or "400" in detail:
                        detail = (
                            f"STT connect 400 for {language} — retrying next language"
                        )
                    await self.send_json({"type": "info", "detail": detail})
                    call_log.error(
                        "STT",
                        detail,
                        session_id=self._sid(),
                        extra={"language": language},
                    )
                    continue
                break

            if self._closed:
                return
            if not connected:
                await self.send_json(
                    {"type": "info", "detail": "STT reconnecting in 1s…"}
                )
            await asyncio.sleep(1.0)

    async def _run_deepgram_session(self, language: str, endpointing: int) -> None:
        api_key = self.settings.deepgram_stt_keys.pick()
        dg = AsyncDeepgramClient(api_key=api_key)
        live_lang = "en" if language.lower().startswith("en-in") else language

        async with dg.listen.v1.connect(
            model=self.settings.deepgram_stt.model,
            language=live_lang,
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            punctuate=True,
            interim_results=True,
            endpointing=max(10, int(endpointing)),
            utterance_end_ms="1000",
            vad_events=True,
            smart_format=True,
        ) as connection:
            self._dg = connection
            connection.on(EventType.MESSAGE, self._on_dg_message)
            connection.on(EventType.ERROR, self._on_dg_error)

            await self.send_json(
                {"type": "info", "detail": f"Listening live ({live_lang})…"}
            )

            listen_task = asyncio.create_task(connection.start_listening())
            pump_task = asyncio.create_task(self._pump_audio(connection))
            keep_task = asyncio.create_task(self._keepalive(connection))

            try:
                done, pending = await asyncio.wait(
                    {listen_task, pump_task, keep_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    exc = task.exception() if not task.cancelled() else None
                    if exc:
                        raise exc
            finally:
                self._dg = None
                try:
                    await connection.send_close_stream()
                except Exception:
                    pass

    async def _keepalive(self, connection: Any) -> None:
        while not self._closed:
            await asyncio.sleep(4.0)
            try:
                await connection.send_keep_alive()
            except Exception:
                break

    async def _pump_audio(self, connection: Any) -> None:
        while not self._closed:
            chunk = await self._audio_q.get()
            if chunk is None:
                break
            try:
                await connection.send_media(chunk)
            except Exception:
                break

    async def _on_dg_error(self, error: Exception) -> None:
        call_log.error("STT", str(error), session_id=self._sid())
        await self.send_json({"type": "info", "detail": f"STT: {error}"})

    def _cancel_spec(self) -> None:
        self._spec_token += 1
        self._spec = None
        if self._spec_task and not self._spec_task.done():
            self._spec_task.cancel()
        if self._spec_debounce and not self._spec_debounce.done():
            self._spec_debounce.cancel()

    def _coach_busy(self) -> bool:
        return self._speaking or time.perf_counter() < self._speak_until

    def _hold_playback(self, text: str = "", pcm_bytes: int = 0) -> None:
        """Keep _speaking True while browser still plays flushed PCM (barge still works)."""
        ms = int(pcm_bytes / 32) if pcm_bytes else max(700, len((text or "").split()) * 260)
        hold = min(ms / 1000.0, 7.0) + 0.2
        self._speaking = True
        self._speak_until = time.perf_counter() + hold
        if self._hold_task and not self._hold_task.done():
            self._hold_task.cancel()

        async def _release() -> None:
            await asyncio.sleep(hold)
            if time.perf_counter() >= self._speak_until - 0.05:
                self._speaking = False

        self._hold_task = asyncio.create_task(_release())

    def _kick_speculative(self, text: str) -> None:
        """Progressive embed: as learner speaks, rebuild reply from live STT text."""
        words = text.split()
        if len(words) < 3:
            return
        if self._coach_busy():
            return
        if self._turn_task and not self._turn_task.done():
            return
        # Clarification / tiny replies: wait for final, don't guess early
        if needs_fresh_reply(text) or is_low_content_turn(text):
            return

        # Identical line — keep current buffer
        if self._spec and _norm(self._spec.source) == _norm(text):
            return

        # Don't thrash: only rebuild after +4 new words (was +2)
        if self._spec and _norm(text).startswith(_norm(self._spec.source)):
            old_n = len(self._spec.source.split())
            grew = len(words) - old_n
            if grew < 4:
                return
            if self._spec_task and not self._spec_task.done():
                return

        async def _debounced() -> None:
            await asyncio.sleep(0.22)
            if self._closed or self._coach_busy():
                return
            if self._turn_task and not self._turn_task.done():
                return
            if needs_fresh_reply(text) or is_low_content_turn(text):
                return
            live = text
            token = self._spec_token + 1
            self._spec_token = token
            if self._spec_task and not self._spec_task.done():
                self._spec_task.cancel()
            self._spec = SpecPipeline(source=live, token=token)
            self._spec_task = asyncio.create_task(self._run_spec(live, token))
            self._dbg(f"spec embed token={token} text={live[:60]!r}")

        if self._spec_debounce and not self._spec_debounce.done():
            self._spec_debounce.cancel()
        self._spec_debounce = asyncio.create_task(_debounced())

    async def _run_spec(self, text: str, token: int) -> None:
        """Loop: draft LLM → stream Deepgram PCM into buffer (ready before final)."""
        try:
            await self.send_json({"type": "prep", "text": text})
            self._dbg(f"spec llm start token={token}")
            draft = await asyncio.to_thread(self._draft_reply, text)
            if token != self._spec_token or self._closed:
                self._dbg(f"spec llm stale token={token} now={self._spec_token}")
                return

            units = split_speak_chunks(draft.coach_text)
            spec = self._spec
            if not spec or spec.token != token:
                return
            spec.user_text = draft.user_text
            spec.coach_text = draft.coach_text
            spec.keywords = draft.keywords
            spec.llm_ms = draft.llm_ms
            spec.units = units
            print(
                f"[lat] PIPE llm ready source={text!r} "
                f"llm={draft.llm_ms}ms units={len(units)} "
                f"coach={draft.coach_text!r}"
            )
            self._dbg(f"spec llm done llm={draft.llm_ms}ms units={len(units)}")
            await self.send_json(
                {
                    "type": "prep_ready",
                    "latency": {
                        "llm_ms": draft.llm_ms,
                        "tts_ms": 0,
                        "prepare_ms": draft.llm_ms,
                        "mode": "pipeline",
                    },
                }
            )

            # Keep Deepgram audio filling while user finishes speaking
            for i, unit in enumerate(units):
                if token != self._spec_token or self._closed:
                    self._dbg(f"spec tts abort before unit {i}")
                    return
                self._dbg(f"spec tts unit {i + 1}/{len(units)} start")
                async for chunk in self.coach.tts.stream_pcm(unit):
                    if token != self._spec_token or self._closed:
                        self._dbg(f"spec tts abort mid unit {i}")
                        return
                    if not self._spec or self._spec.token != token:
                        return
                    self._spec.pcm.extend(chunk)
                    if not self._spec.audio_ready:
                        self._spec.audio_ready = True
                        buffered_ms = int(len(self._spec.pcm) / 32)  # 16k mono s16
                        print(
                            f"[lat] PIPE audio ready buffered~{buffered_ms}ms "
                            f"unit={i + 1}/{len(units)}"
                        )
                        self._dbg(f"spec first audio ~{buffered_ms}ms")
                        await self.send_json(
                            {
                                "type": "prep_audio_ready",
                                "buffered_ms": buffered_ms,
                                "unit": i + 1,
                                "units": len(units),
                            }
                        )
                if self._spec and self._spec.token == token:
                    self._spec.units_done = i + 1
                    self._dbg(
                        f"spec unit done {i + 1}/{len(units)} "
                        f"bytes={len(self._spec.pcm)}"
                    )

            if self._spec and self._spec.token == token:
                self._spec.done = True
                print(
                    f"[lat] PIPE full ready bytes={len(self._spec.pcm)} "
                    f"units={self._spec.units_done}"
                )
                self._dbg("spec pipeline full ready")
        except asyncio.CancelledError:
            self._dbg(f"spec cancelled token={token}")
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[lat] PIPE fail: {exc}")
            self._dbg(f"spec FAIL: {exc}")

    async def _freeze_spec(self, text: str) -> SpecSnap | None:
        """Stop pipeline writes and snapshot buffered LLM + PCM for instant play."""
        spec = self._spec
        if not spec or not transcripts_compatible(spec.source, text):
            self._dbg(
                f"freeze miss compatible={bool(spec)} "
                f"src={(spec.source[:40] if spec else '')!r} final={text[:40]!r}"
            )
            return None

        self._dbg("freeze wait for first unit")
        # After speech_final, let the loop finish first sentence audio if close
        deadline = time.perf_counter() + 2.25
        reason = "timeout"
        while time.perf_counter() < deadline:
            spec = self._spec
            if not spec or not transcripts_compatible(spec.source, text):
                self._dbg("freeze aborted: incompatible mid-wait")
                return None
            if spec.audio_ready and spec.units_done >= 1:
                reason = "unit1_ready"
                break
            if spec.done:
                reason = "done"
                break
            task_done = not self._spec_task or self._spec_task.done()
            if task_done:
                if spec.coach_text:
                    reason = "task_done_llm"
                    break
                self._dbg("freeze aborted: task done without llm")
                return None
            await asyncio.sleep(0.02)

        spec = self._spec
        if not spec or not spec.coach_text:
            self._dbg("freeze empty after wait")
            return None
        if not transcripts_compatible(spec.source, text):
            return None

        self._dbg(
            f"freeze snap reason={reason} units_done={spec.units_done} "
            f"bytes={len(spec.pcm)} audio={spec.audio_ready}"
        )

        # Freeze: invalidate token so _run_spec stops appending
        self._spec_token += 1
        if self._spec_task and not self._spec_task.done():
            self._spec_task.cancel()
            # NEVER await forever — httpx stream cancel can hang the whole turn
            try:
                await asyncio.wait_for(self._spec_task, timeout=0.35)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as exc:
                self._dbg(f"freeze cancel wait end: {type(exc).__name__}")
                if self._spec_task and not self._spec_task.done():
                    self._dbg("WARN orphaned spec task still running after cancel timeout")

        snap = SpecSnap(
            user_text=spec.user_text or text,
            coach_text=spec.coach_text,
            keywords=list(spec.keywords),
            llm_ms=spec.llm_ms,
            units=list(spec.units),
            units_done=spec.units_done,
            pcm=bytes(spec.pcm),
            done=spec.done,
        )
        self._spec = None
        if self._spec_debounce and not self._spec_debounce.done():
            self._spec_debounce.cancel()
        self._dbg(
            f"freeze ok mode={'hit' if snap.pcm and snap.units_done >= 1 else 'llm'} "
            f"bytes={len(snap.pcm)}"
        )
        return snap

    async def _send_pcm_bytes(self, pcm: bytes, generation: int) -> None:
        """Flush buffered PCM to client in small frames."""
        frame = 2048
        for i in range(0, len(pcm), frame):
            if generation != self._generation:
                return
            try:
                await self.ws.send_bytes(pcm[i : i + frame])
            except Exception:
                return
            # Yield so browser can start decoding ASAP
            if i == 0:
                await asyncio.sleep(0)

    async def _stream_units(
        self,
        units: list[str],
        *,
        start_at: int,
        generation: int,
    ) -> int:
        """Stream remaining speak units; return ttfb_ms for first byte (0 if none)."""
        ttfb_ms = 0
        first = True
        t0 = time.perf_counter()
        for i, unit in enumerate(units[start_at:], start=start_at):
            if generation != self._generation:
                return ttfb_ms
            print(f"[tts] remain {i + 1}/{len(units)}: {unit!r}")
            async for chunk in self.coach.tts.stream_pcm(unit):
                if generation != self._generation:
                    return ttfb_ms
                if first:
                    ttfb_ms = int((time.perf_counter() - t0) * 1000)
                    first = False
                try:
                    await self.ws.send_bytes(chunk)
                except Exception:
                    return ttfb_ms
        return ttfb_ms
    async def _barge_in(self, reason: str = "", interrupt_text: str = "") -> None:
        """Stop coach audio when user clearly starts speaking over it."""
        if not self._coach_busy():
            return
        age = time.perf_counter() - self._speak_started_at
        if age < self._barge_grace_s:
            print(f"[call] barge grace ({age:.2f}s) skip {reason}")
            return

        self._generation += 1
        self._speaking = False
        self._speak_until = 0.0
        if self._hold_task and not self._hold_task.done():
            self._hold_task.cancel()
        self._cancel_spec()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        print(f"[call] BARGE_IN {reason}")
        await self.send_json({"type": "barge_in"})

        text = (interrupt_text or "").strip()
        if text:
            self._pending = text
            if self._barge_task and not self._barge_task.done():
                self._barge_task.cancel()

            async def _followup() -> None:
                await asyncio.sleep(0.3)
                if not self._closed and self._pending and not self._speaking:
                    self._schedule_turn()

            self._barge_task = asyncio.create_task(_followup())

    async def _on_dg_message(self, message: Any) -> None:
        if isinstance(message, ListenV1SpeechStarted):
            return

        if isinstance(message, ListenV1Results):
            text = _transcript_from_results(message)
            if text:
                self._touch_voice()
                conf = _confidence_from_results(message)
                if conf is not None:
                    self._last_stt_confidence = conf
                await self.send_json(
                    {
                        "type": "partial",
                        "text": text,
                        "is_final": bool(message.is_final),
                        "speech_final": bool(message.speech_final),
                    }
                )
                self._last_partial = text
                if message.is_final or message.speech_final or self._speaking:
                    self._dbg(
                        f"stt text={text[:50]!r} final={bool(message.is_final)} "
                        f"speech_final={bool(message.speech_final)} "
                        f"speaking={self._speaking}"
                    )

                if message.is_final:
                    self._pending = merge_pending_stt(self._pending, text)
                    self._dbg(f"stt is_final pending={self._pending[:60]!r}")

                if self._coach_busy():
                    words = text.split()
                    if (
                        (message.is_final or message.speech_final)
                        and len(words) >= self._barge_min_words
                    ):
                        await self._barge_in(
                            f"final={text!r}",
                            interrupt_text=self._pending or text,
                        )
                    return

                if not message.speech_final:
                    live = text
                    if self._pending:
                        live = self._pending
                    self._kick_speculative(live)
                if message.speech_final and self._pending:
                    self._dbg(f"stt speech_final -> schedule pending={self._pending[:60]!r}")
                    self._schedule_turn()
            return

        if (
            isinstance(message, ListenV1UtteranceEnd)
            and self._pending
            and not self._coach_busy()
        ):
            self._dbg(f"stt utterance_end -> schedule pending={self._pending[:60]!r}")
            self._schedule_turn()

    def _awaiting_correction(self) -> bool:
        rt = self.conversation_runtime
        cid = getattr(self.session, "conversation_id", None) or ""
        if rt is None or not cid:
            return False
        check = getattr(rt, "is_awaiting_correction", None)
        if not callable(check):
            return False
        try:
            return bool(check(cid))
        except Exception:
            return False

    def _schedule_turn(self) -> None:
        text = self._pending.strip()
        self._pending = ""
        if not text:
            self._dbg("schedule skip empty")
            return
        if is_low_content_turn(text) and not self._awaiting_correction():
            self._persist_message(
                "user",
                text,
                {"source": "voice", "sttProvider": "deepgram", "skipped": "ack"},
            )
            print(f"[call] skip ack (no LLM): {text!r}")
            self._dbg(f"schedule skip ack: {text[:60]!r}")
            return

        if self._turn_task and not self._turn_task.done() and not self._speaking:
            self._queued = f"{self._queued} {text}".strip()
            print(f"[call] queued while thinking: {text!r}")
            self._dbg(f"QUEUED while thinking: {text[:60]!r}")
            return

        if self._turn_task and not self._turn_task.done():
            self._dbg("cancel previous turn for barge/new")
            self._turn_task.cancel()

        self._generation += 1
        gen = self._generation
        self._turn_started_at = time.perf_counter()
        print(f"[call] turn start: {text!r}")
        self._dbg(f"TURN START gen={gen} text={text[:60]!r}")
        self._turn_task = asyncio.create_task(self._run_turn(text, gen))

    async def _run_turn(self, text: str, generation: int) -> None:
        t_end = time.perf_counter()
        await self.send_json({"type": "user_final", "text": text})
        self._persist_message(
            "user",
            text,
            {"source": "voice", "sttProvider": "deepgram"},
        )
        await self.send_json({"type": "thinking"})
        self._dbg(f"turn thinking gen={generation}")
        try:
            # Clarifications must hit a fresh LLM answer (no wrong speculative story)
            snap = None
            if needs_fresh_reply(text):
                self._cancel_spec()
                self._dbg("turn force fresh (clarify/short)")
            else:
                snap = await self._freeze_spec(text)
                if snap is None:
                    self._cancel_spec()
                    self._dbg("turn no snap -> cold path")
                else:
                    self._dbg(
                        f"turn snap units_done={snap.units_done} "
                        f"pcm={len(snap.pcm)} done={snap.done}"
                    )

            # Short user line + rambling 4-unit buffer ("No. No.") → first unit only
            if (
                snap is not None
                and len(text.split()) <= 3
                and len(snap.units) > 2
            ):
                snap.units = snap.units[:1]
                snap.coach_text = snap.units[0]
                snap.pcm = b""
                snap.units_done = 0
                snap.done = True
                self._dbg("turn cap ramble: short user, first unit only")

            if snap is not None and snap.pcm and snap.units_done >= 1:
                # Pipeline HIT — at least one speak-unit fully buffered; flush now
                wait_pre = int((time.perf_counter() - t_end) * 1000)
                turn = self._commit_turn(
                    user_text=snap.user_text,
                    coach_text=snap.coach_text,
                    keywords=snap.keywords,
                )
                self._persist_message(
                    "assistant",
                    snap.coach_text,
                    {"source": "ai", "ttsProvider": "deepgram"},
                )
                self._speaking = True
                self._speak_started_at = time.perf_counter()
                try:
                    await self.send_json(
                        {
                            "type": "coach_turn",
                            "turn": turn,
                            "user_text": snap.user_text,
                            "coach_text": snap.coach_text,
                            "keywords": snap.keywords,
                            "stream": True,
                            "audio_format": "pcm_s16le",
                            "sample_rate": 16000,
                            "mode": "pipeline_hit",
                        }
                    )
                    t_tts = time.perf_counter()
                    await self._send_pcm_bytes(snap.pcm, generation)
                    flush_ms = int((time.perf_counter() - t_tts) * 1000)
                    print(
                        f"[lat] PIPE HIT flush={flush_ms}ms pre={wait_pre}ms "
                        f"llm={snap.llm_ms}ms bytes={len(snap.pcm)} "
                        f"units_done={snap.units_done}/{len(snap.units)} "
                        f"coach={snap.coach_text!r}"
                    )
                    remain_ttfb = 0
                    if not snap.done and snap.units_done < len(snap.units):
                        remain_ttfb = await self._stream_units(
                            snap.units,
                            start_at=snap.units_done,
                            generation=generation,
                        )
                    tts_total_ms = int((time.perf_counter() - t_tts) * 1000)
                    heard_after = wait_pre  # audio was pre-buffered
                    latency = {
                        "llm_ms": snap.llm_ms,
                        "tts_ms": tts_total_ms,
                        "ttfb_ms": 0,
                        "flush_ms": flush_ms,
                        "remain_ttfb_ms": remain_ttfb,
                        "prepare_ms": snap.llm_ms,
                        "wait_ms": heard_after,
                        "total_ms": heard_after,
                        "speculative": True,
                        "mode": "pipeline_hit",
                    }
                    self._log_lat(latency)
                    await self.send_json(
                        {"type": "coach_audio_end", "latency": latency}
                    )
                    self._dbg(f"turn PIPE HIT done hear@{heard_after}ms")
                finally:
                    if generation == self._generation:
                        self._hold_playback(snap.coach_text, pcm_bytes=len(snap.pcm))
                    else:
                        self._speaking = False

            elif snap is not None and snap.coach_text:
                # LLM ready, audio not yet — stream TTS now
                wait_pre = int((time.perf_counter() - t_end) * 1000)
                turn = self._commit_turn(
                    user_text=snap.user_text,
                    coach_text=snap.coach_text,
                    keywords=snap.keywords,
                )
                self._persist_message(
                    "assistant",
                    snap.coach_text,
                    {"source": "ai", "ttsProvider": "deepgram"},
                )
                self._speaking = True
                self._speak_started_at = time.perf_counter()
                ttfb_ms = 0
                try:
                    await self.send_json(
                        {
                            "type": "coach_turn",
                            "turn": turn,
                            "user_text": snap.user_text,
                            "coach_text": snap.coach_text,
                            "keywords": snap.keywords,
                            "stream": True,
                            "audio_format": "pcm_s16le",
                            "sample_rate": 16000,
                            "mode": "pipeline_llm",
                        }
                    )
                    t_tts = time.perf_counter()
                    first = True
                    async for chunk in self.coach.tts.stream_pcm_chunked(
                        snap.coach_text
                    ):
                        if generation != self._generation:
                            return
                        if first:
                            ttfb_ms = int((time.perf_counter() - t_tts) * 1000)
                            first = False
                            print(
                                f"[lat] PIPE LLM-only first ttfb={ttfb_ms}ms "
                                f"pre={wait_pre}ms llm={snap.llm_ms}ms"
                            )
                        try:
                            await self.ws.send_bytes(chunk)
                        except Exception:
                            break
                    tts_total_ms = int((time.perf_counter() - t_tts) * 1000)
                    heard_after = wait_pre + ttfb_ms
                    latency = {
                        "llm_ms": snap.llm_ms,
                        "tts_ms": tts_total_ms,
                        "ttfb_ms": ttfb_ms,
                        "prepare_ms": snap.llm_ms,
                        "wait_ms": heard_after,
                        "total_ms": heard_after,
                        "speculative": True,
                        "mode": "pipeline_llm",
                    }
                    self._log_lat(latency)
                    await self.send_json(
                        {"type": "coach_audio_end", "latency": latency}
                    )
                    self._dbg(f"turn PIPE LLM done hear@{heard_after}ms")
                finally:
                    if generation == self._generation:
                        self._hold_playback(snap.coach_text)
                    else:
                        self._speaking = False
            else:
                # Cold path: LLM then sentence-chunk TTS
                self._dbg("turn cold LLM start")
                draft = await asyncio.to_thread(self._draft_reply, text)
                self._dbg(f"turn cold LLM done llm={draft.llm_ms}ms")
                if generation != self._generation:
                    return

                turn = self._commit_turn(
                    user_text=draft.user_text,
                    coach_text=draft.coach_text,
                    keywords=draft.keywords,
                )
                self._persist_message(
                    "assistant",
                    draft.coach_text,
                    {"source": "ai", "ttsProvider": "deepgram"},
                )
                self._speaking = True
                self._speak_started_at = time.perf_counter()
                ttfb_ms = 0
                try:
                    await self.send_json(
                        {
                            "type": "coach_turn",
                            "turn": turn,
                            "user_text": draft.user_text,
                            "coach_text": draft.coach_text,
                            "keywords": draft.keywords,
                            "stream": True,
                            "audio_format": "pcm_s16le",
                            "sample_rate": 16000,
                            "mode": "fresh",
                        }
                    )
                    t_tts = time.perf_counter()
                    first = True
                    async for chunk in self.coach.tts.stream_pcm_chunked(
                        draft.coach_text
                    ):
                        if generation != self._generation:
                            return
                        if first:
                            ttfb_ms = int((time.perf_counter() - t_tts) * 1000)
                            wait_ms = int((time.perf_counter() - t_end) * 1000)
                            first = False
                            print(
                                f"[lat] FIRST_AUDIO wait={wait_ms}ms "
                                f"llm={draft.llm_ms}ms ttfb={ttfb_ms}ms "
                                f"coach={draft.coach_text!r}"
                            )
                        try:
                            await self.ws.send_bytes(chunk)
                        except Exception:
                            break
                    tts_total_ms = int((time.perf_counter() - t_tts) * 1000)
                    heard_after = draft.llm_ms + ttfb_ms
                    latency = {
                        "llm_ms": draft.llm_ms,
                        "tts_ms": tts_total_ms,
                        "ttfb_ms": ttfb_ms,
                        "prepare_ms": draft.llm_ms + tts_total_ms,
                        "wait_ms": heard_after,
                        "total_ms": heard_after,
                        "speculative": False,
                        "mode": "fresh",
                    }
                    self._log_lat(latency)
                    await self.send_json(
                        {"type": "coach_audio_end", "latency": latency}
                    )
                    print(
                        f"[lat] STREAM done ttfb={ttfb_ms}ms "
                        f"tts_total={tts_total_ms}ms heard_after={heard_after}ms"
                    )
                    self._dbg(f"turn FRESH done hear@{heard_after}ms")
                finally:
                    if generation == self._generation:
                        self._hold_playback(draft.coach_text)
                    else:
                        self._speaking = False

            if self._queued.strip() and generation == self._generation:
                queued = self._queued.strip()
                self._queued = ""
                if is_low_content_turn(queued) and not self._awaiting_correction():
                    self._persist_message(
                        "user",
                        queued,
                        {"source": "voice", "sttProvider": "deepgram", "skipped": "ack"},
                    )
                    self._dbg(f"drain skip ack: {queued[:60]!r}")
                    self._dbg("turn idle listening")
                    return
                wait = max(0.0, self._speak_until - time.perf_counter())
                self._dbg(f"drain queued in {wait:.2f}s -> {queued[:60]!r}")

                async def _drain_later() -> None:
                    await asyncio.sleep(wait)
                    if self._closed or generation != self._generation:
                        return
                    self._pending = queued
                    self._schedule_turn()

                asyncio.create_task(_drain_later())
            else:
                self._dbg("turn idle listening")
        except asyncio.CancelledError:
            self._speaking = False
            self._speak_until = 0.0
            self._dbg(f"turn CANCELLED gen={generation}")
            raise
        except Exception as exc:  # noqa: BLE001
            self._speaking = False
            self._speak_until = 0.0
            print(f"[call] turn error: {exc}")
            self._dbg(f"turn ERROR: {exc}")
            if generation == self._generation:
                await self.send_json({"type": "error", "detail": str(exc)})

    async def _send_full_audio(self, result: Any, latency: dict[str, Any]) -> None:
        meta = {
            "type": "coach_turn",
            "turn": result.turn,
            "user_text": result.user_text,
            "coach_text": result.coach_text,
            "keywords": result.keywords,
            "latency": latency,
            "audio_format": "wav",
        }
        if (
            result.coach_audio_bytes
            and self.ws.client_state == WebSocketState.CONNECTED
        ):
            await self.send_json({**meta, "audio_binary_next": True})
            try:
                await self.ws.send_bytes(result.coach_audio_bytes)
            except Exception:
                await self.send_json(
                    {
                        **meta,
                        "coach_audio_b64": base64.b64encode(
                            result.coach_audio_bytes
                        ).decode("ascii"),
                    }
                )
        else:
            await self.send_json(meta)
