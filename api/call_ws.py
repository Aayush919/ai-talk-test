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
    needs_fresh_reply,
    transcripts_compatible,
)
from core import call_log
from core.config import Settings
from core.session import Session
from core.text_clean import safe_print
from wrappers.deepgram_tts import split_speak_chunks

# local import of _norm via needs / compatible already enough
from core.coach_service import _norm

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
    ) -> None:
        self.ws = websocket
        self.session = session
        self.coach = coach
        self.settings = settings
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
        self._dg = None
        self._last_partial = ""
        self._spec_debounce: asyncio.Task[None] | None = None
        # Soft barge-in: ignore echo for a moment, then allow real interrupt
        self._barge_grace_s = 0.85
        self._barge_min_words = 2
        self._barge_task: asyncio.Task[None] | None = None
        self._last_event = "init"
        self._turn_started_at = 0.0

    def _sid(self) -> str:
        return getattr(self.session, "session_id", "") or ""

    def _dbg(self, msg: str) -> None:
        """Console + logs/ai-talk.log + logs/call-<sid>.log"""
        self._last_event = msg
        level = "error" if any(
            w in msg.upper() for w in ("FAIL", "ERROR", "STUCK", "WARN")
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
                "ttfb_ms": latency.get("ttfb_ms"),
                "tts_ms": latency.get("tts_ms"),
                "flush_ms": latency.get("flush_ms"),
                "spec": latency.get("speculative"),
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

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.ws.client_state != WebSocketState.CONNECTED or self._closed:
            return
        try:
            await self.ws.send_json(payload)
        except Exception:
            pass

    async def run(self) -> None:
        await self.ws.accept()
        await self.send_json(
            {
                "type": "call_ready",
                "session_id": self.session.session_id,
                "message": "Live call connected — speak anytime.",
            }
        )

        recv_task = asyncio.create_task(self._recv_browser(), name="recv")
        dg_task = asyncio.create_task(self._deepgram_loop(), name="deepgram")
        hb_task = asyncio.create_task(self._heartbeat(), name="heartbeat")
        call_log.info(
            "CONNECT",
            "live call connected",
            session_id=self._sid(),
            extra={
                "topic": getattr(self.session.topic, "id", "") if self.session.topic else "",
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
            await self._audio_q.put(None)
            self._cancel_spec()
            if self._turn_task and not self._turn_task.done():
                self._turn_task.cancel()
            hb_task.cancel()
            dg_task.cancel()
            await asyncio.gather(dg_task, hb_task, return_exceptions=True)
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

    def _kick_speculative(self, text: str) -> None:
        """Progressive embed: as learner speaks, rebuild reply from live STT text."""
        words = text.split()
        if len(words) < 2:
            return
        if self._speaking:
            return
        if self._turn_task and not self._turn_task.done():
            return
        # Clarification / tiny replies: wait for final, don't guess early
        if needs_fresh_reply(text):
            return

        # Identical line — keep current buffer
        if self._spec and _norm(self._spec.source) == _norm(text):
            return

        # Progressive rebuild: every +2 new words, re-embed from latest speech
        if self._spec and _norm(text).startswith(_norm(self._spec.source)):
            old_n = len(self._spec.source.split())
            new_n = len(words)
            grew = new_n - old_n
            if grew < 2:
                return
            # If first audio unit almost ready and only +2 words, let it finish
            if (
                grew <= 2
                and self._spec.audio_ready
                and not self._spec.done
                and self._spec_task
                and not self._spec_task.done()
            ):
                return

        async def _debounced() -> None:
            await asyncio.sleep(0.10)
            if self._closed or self._speaking:
                return
            if self._turn_task and not self._turn_task.done():
                return
            if needs_fresh_reply(text):
                return
            # Capture latest pending-ish text at fire time
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
            draft = await asyncio.to_thread(
                self.coach.draft_reply, self.session, text
            )
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
        deadline = time.perf_counter() + 1.15
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
        if not self._speaking:
            return
        age = time.perf_counter() - self._speak_started_at
        if age < self._barge_grace_s:
            print(f"[call] barge grace ({age:.2f}s) skip {reason}")
            return

        self._generation += 1
        self._speaking = False
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

                # Interrupt only on committed speech (not noisy partials / echo)
                if self._speaking:
                    words = text.split()
                    if (
                        (message.is_final or message.speech_final)
                        and len(words) >= self._barge_min_words
                    ):
                        await self._barge_in(
                            f"final={text!r}",
                            interrupt_text=text,
                        )
                    return

                if not message.speech_final:
                    live = f"{self._pending} {text}".strip() if self._pending else text
                    self._kick_speculative(live)

                if message.is_final:
                    self._pending = f"{self._pending} {text}".strip()
                    self._dbg(f"stt is_final pending={self._pending[:60]!r}")
                if message.speech_final and self._pending:
                    self._dbg(f"stt speech_final -> schedule pending={self._pending[:60]!r}")
                    self._schedule_turn()
            return

        if (
            isinstance(message, ListenV1UtteranceEnd)
            and self._pending
            and not self._speaking
        ):
            self._dbg(f"stt utterance_end -> schedule pending={self._pending[:60]!r}")
            self._schedule_turn()

    def _schedule_turn(self) -> None:
        text = self._pending.strip()
        self._pending = ""
        if not text:
            self._dbg("schedule skip empty")
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

            if snap is not None and snap.pcm and snap.units_done >= 1:
                # Pipeline HIT — at least one speak-unit fully buffered; flush now
                wait_pre = int((time.perf_counter() - t_end) * 1000)
                turn = self.coach.commit_text(
                    self.session,
                    user_text=snap.user_text,
                    coach_text=snap.coach_text,
                    keywords=snap.keywords,
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
                    self._speaking = False

            elif snap is not None and snap.coach_text:
                # LLM ready, audio not yet — stream TTS now
                wait_pre = int((time.perf_counter() - t_end) * 1000)
                turn = self.coach.commit_text(
                    self.session,
                    user_text=snap.user_text,
                    coach_text=snap.coach_text,
                    keywords=snap.keywords,
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
                    self._speaking = False
            else:
                # Cold path: LLM then sentence-chunk TTS
                self._dbg("turn cold LLM start")
                draft = await asyncio.to_thread(
                    self.coach.draft_reply, self.session, text
                )
                self._dbg(f"turn cold LLM done llm={draft.llm_ms}ms")
                if generation != self._generation:
                    return

                turn = self.coach.commit_text(
                    self.session,
                    user_text=draft.user_text,
                    coach_text=draft.coach_text,
                    keywords=draft.keywords,
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
                    self._speaking = False

            if self._queued.strip() and generation == self._generation:
                queued = self._queued.strip()
                self._queued = ""
                self._pending = queued
                self._dbg(f"drain queued -> {queued[:60]!r}")
                self._schedule_turn()
            else:
                self._dbg("turn idle listening")
        except asyncio.CancelledError:
            self._speaking = False
            self._dbg(f"turn CANCELLED gen={generation}")
            raise
        except Exception as exc:  # noqa: BLE001
            self._speaking = False
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
