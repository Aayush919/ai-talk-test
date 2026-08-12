"""Realtime call — speculative prep + streaming TTS for low TTFB."""

from __future__ import annotations

import asyncio
import base64
import json
import time
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
    PreparedReply,
    transcripts_compatible,
)
from core.config import Settings
from core.session import Session


def _transcript_from_results(message: ListenV1Results) -> str:
    channel = message.channel
    alts = getattr(channel, "alternatives", None) or []
    if not alts:
        return ""
    return (getattr(alts[0], "transcript", None) or "").strip()


class LiveCallBridge:
    """
    While user speaks: speculative Groq+TTS on partials.
    Fresh turns: stream TTS PCM so playback starts on first chunk (not full file).
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
        self._spec_ready: PreparedReply | None = None
        self._spec_source = ""
        self._spec_token = 0
        self._generation = 0
        self._closed = False
        self._speaking = False
        self._speak_started_at = 0.0
        self._dg = None
        self._last_partial = ""
        self._spec_debounce: asyncio.Task[None] | None = None
        # Ignore echo / noise barge-ins for this long after coach audio starts
        self._barge_grace_s = 1.4
        self._barge_min_words = 3

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

        try:
            await recv_task
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            await self.send_json({"type": "error", "detail": str(exc)})
        finally:
            self._closed = True
            await self._audio_q.put(None)
            self._cancel_spec()
            if self._turn_task and not self._turn_task.done():
                self._turn_task.cancel()
            dg_task.cancel()
            await asyncio.gather(dg_task, return_exceptions=True)
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
            if payload.get("type") == "end_call":
                break
            if payload.get("type") == "ping":
                await self.send_json({"type": "pong"})

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
        await self.send_json({"type": "info", "detail": f"STT: {error}"})

    def _cancel_spec(self) -> None:
        self._spec_token += 1
        self._spec_ready = None
        self._spec_source = ""
        if self._spec_task and not self._spec_task.done():
            self._spec_task.cancel()
        if self._spec_debounce and not self._spec_debounce.done():
            self._spec_debounce.cancel()

    def _kick_speculative(self, text: str) -> None:
        words = text.split()
        if len(words) < 3:
            return
        if self._speaking:
            return
        if self._turn_task and not self._turn_task.done():
            return

        if self._spec_source and transcripts_compatible(self._spec_source, text):
            if abs(len(text) - len(self._spec_source)) < 8 and self._spec_ready:
                return

        async def _debounced() -> None:
            await asyncio.sleep(0.18)
            if self._closed or self._speaking:
                return
            token = self._spec_token + 1
            self._spec_token = token
            self._spec_source = text
            self._spec_ready = None
            if self._spec_task and not self._spec_task.done():
                self._spec_task.cancel()
            self._spec_task = asyncio.create_task(self._run_spec(text, token))

        if self._spec_debounce and not self._spec_debounce.done():
            self._spec_debounce.cancel()
        self._spec_debounce = asyncio.create_task(_debounced())

    async def _run_spec(self, text: str, token: int) -> None:
        try:
            await self.send_json({"type": "prep", "text": text})
            prepared = await asyncio.to_thread(
                self.coach.prepare_reply,
                self.session,
                text,
                speculative=True,
            )
            if token != self._spec_token or self._closed:
                return
            self._spec_ready = prepared
            print(
                f"[lat] SPEC ready source={text!r} "
                f"llm={prepared.llm_ms}ms tts={prepared.tts_ms}ms "
                f"total={prepared.prepare_ms}ms"
            )
            await self.send_json(
                {
                    "type": "prep_ready",
                    "latency": {
                        "llm_ms": prepared.llm_ms,
                        "tts_ms": prepared.tts_ms,
                        "prepare_ms": prepared.prepare_ms,
                    },
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[lat] SPEC fail: {exc}")

    async def _barge_in(self, reason: str = "") -> None:
        """Interrupt coach audio only for real user speech (not speaker echo)."""
        if not self._speaking:
            return
        age = time.perf_counter() - self._speak_started_at
        if age < self._barge_grace_s:
            print(f"[call] barge ignored (grace {age:.2f}s) {reason}")
            return
        self._generation += 1
        self._speaking = False
        self._cancel_spec()
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        print(f"[call] barge_in {reason}")
        await self.send_json({"type": "barge_in"})

    async def _on_dg_message(self, message: Any) -> None:
        # SpeechStarted alone is too noisy (echo) — never barge on it
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

                # Only barge on committed speech with enough words
                if self._speaking and message.is_final and len(text.split()) >= self._barge_min_words:
                    await self._barge_in(f"final={text!r}")

                if not message.speech_final and not self._speaking:
                    live = f"{self._pending} {text}".strip() if self._pending else text
                    self._kick_speculative(live)

                if message.is_final:
                    self._pending = f"{self._pending} {text}".strip()
                if message.speech_final and self._pending and not self._speaking:
                    self._schedule_turn()
            return

        if (
            isinstance(message, ListenV1UtteranceEnd)
            and self._pending
            and not self._speaking
        ):
            self._schedule_turn()

    def _schedule_turn(self) -> None:
        text = self._pending.strip()
        self._pending = ""
        if not text:
            return

        if self._turn_task and not self._turn_task.done() and not self._speaking:
            self._queued = f"{self._queued} {text}".strip()
            print(f"[call] queued while thinking: {text!r}")
            return

        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()

        self._generation += 1
        gen = self._generation
        print(f"[call] turn start: {text!r}")
        self._turn_task = asyncio.create_task(self._run_turn(text, gen))

    async def _run_turn(self, text: str, generation: int) -> None:
        t_end = time.perf_counter()
        await self.send_json({"type": "user_final", "text": text})
        await self.send_json({"type": "thinking"})
        try:
            prepared: PreparedReply | None = None

            if (
                self._spec_ready
                and self._spec_source
                and transcripts_compatible(self._spec_source, text)
            ):
                prepared = self._spec_ready
                prepared.speculative = True
                print(f"[lat] HIT speculative for {text!r}")
            elif self._spec_task and not self._spec_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(self._spec_task), timeout=0.8)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                if (
                    generation == self._generation
                    and self._spec_ready
                    and transcripts_compatible(self._spec_source, text)
                ):
                    prepared = self._spec_ready
                    prepared.speculative = True
                    print(f"[lat] HIT speculative after wait for {text!r}")

            self._cancel_spec()

            if prepared is not None and prepared.audio:
                wait_ms = int((time.perf_counter() - t_end) * 1000)
                result = self.coach.commit_prepared(
                    self.session, prepared, wait_ms=wait_ms
                )
                latency = {
                    "llm_ms": prepared.llm_ms,
                    "tts_ms": prepared.tts_ms,
                    "ttfb_ms": 0,
                    "prepare_ms": prepared.prepare_ms,
                    "wait_ms": wait_ms,
                    "total_ms": wait_ms,
                    "speculative": True,
                    "mode": "spec_buffer",
                }
                print(
                    f"[lat] SEND wait={wait_ms}ms llm={prepared.llm_ms}ms "
                    f"tts={prepared.tts_ms}ms SPEC HIT coach={result.coach_text!r}"
                )
                self._speaking = True
                self._speak_started_at = time.perf_counter()
                try:
                    await self._send_full_audio(result, latency)
                finally:
                    self._speaking = False
            else:
                # Fresh: LLM then stream TTS — hear on first chunk
                draft = await asyncio.to_thread(
                    self.coach.draft_reply, self.session, text
                )
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
                        }
                    )
                    t_tts = time.perf_counter()
                    first = True
                    async for chunk in self.coach.tts.stream_pcm(draft.coach_text):
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
                        "mode": "stream",
                    }
                    await self.send_json(
                        {"type": "coach_audio_end", "latency": latency}
                    )
                    print(
                        f"[lat] STREAM done ttfb={ttfb_ms}ms "
                        f"tts_total={tts_total_ms}ms heard_after={heard_after}ms"
                    )
                finally:
                    self._speaking = False

            if self._queued.strip() and generation == self._generation:
                queued = self._queued.strip()
                self._queued = ""
                self._pending = queued
                self._schedule_turn()
        except asyncio.CancelledError:
            self._speaking = False
            raise
        except Exception as exc:  # noqa: BLE001
            self._speaking = False
            print(f"[call] turn error: {exc}")
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
