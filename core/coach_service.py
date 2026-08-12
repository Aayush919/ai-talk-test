"""Live coach — prepare (speculative) + commit, with latency timings."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from core.session import Session
from core.tfidf_engine import TfidfEngine
from wrappers.deepgram_tts import DeepgramTTS
from wrappers.groq_llm import GroqCoach
from wrappers.mongo_store import MongoStore


@dataclass
class PreparedReply:
    user_text: str
    coach_text: str
    keywords: list[str]
    audio: bytes
    llm_ms: int
    tts_ms: int
    prepare_ms: int
    speculative: bool = False
    ttfb_ms: int = 0


@dataclass
class DraftReply:
    user_text: str
    coach_text: str
    keywords: list[str]
    llm_ms: int


@dataclass
class TurnResult:
    turn: int
    user_text: str
    coach_text: str
    keywords: list[str]
    coach_audio_bytes: bytes | None = None
    latency: dict[str, int] | None = None
    used_speculative: bool = False


def _norm(text: str) -> str:
    raw = (text or "").lower()
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in raw)
    return " ".join(cleaned.split())


def transcripts_compatible(partial: str, final: str) -> bool:
    """Reuse speculative prep when final is close enough to what we prepared on."""
    a, b = _norm(partial), _norm(final)
    if not a or not b:
        return False
    if a == b:
        return True
    if b.startswith(a) or a.startswith(b):
        return True
    wa, wb = a.split(), b.split()
    if len(wa) >= 3 and " ".join(wa[:3]) == " ".join(wb[:3]):
        return True
    if len(wa) >= 4 and " ".join(wa[-4:]) in b:
        return True
    return False


class CoachService:
    def __init__(
        self,
        *,
        tts: DeepgramTTS,
        coach: GroqCoach,
        tfidf: TfidfEngine,
        mongo: MongoStore,
    ) -> None:
        self.tts = tts
        self.coach = coach
        self.tfidf = tfidf
        self.mongo = mongo

    def open_session(self, session: Session) -> TurnResult:
        starter = (
            session.topic.starter
            if session.topic
            else (
                "Hi! I'm your English talk coach. "
                "What would you like to practice today?"
            )
        )
        session.add("assistant", starter)
        session.turn = 0
        t0 = time.perf_counter()
        audio = self.tts.speak_bytes(starter)
        tts_ms = int((time.perf_counter() - t0) * 1000)
        self._persist_async(
            session.session_id,
            turn=0,
            user_text=None,
            coach_text=starter,
            keywords=[],
        )
        return TurnResult(
            turn=0,
            user_text="",
            coach_text=starter,
            keywords=[],
            coach_audio_bytes=audio,
            latency={"tts_ms": tts_ms, "total_ms": tts_ms},
        )

    def draft_reply(self, session: Session, user_text: str) -> DraftReply:
        """LLM only — TTS streams separately for lower time-to-first-audio."""
        text = (user_text or "").strip()
        history_texts = [m["content"] for m in session.messages]
        keywords = self.tfidf.extract(history_texts, text)
        history = session.snapshot() + [{"role": "user", "content": text}]
        t_llm = time.perf_counter()
        coach_text = (
            self.coach.reply(history, keywords, topic=session.topic) or ""
        ).strip()
        if not coach_text:
            coach_text = "Nice — tell me a bit more. What else?"
        return DraftReply(
            user_text=text,
            coach_text=coach_text,
            keywords=keywords,
            llm_ms=int((time.perf_counter() - t_llm) * 1000),
        )

    def prepare_reply(
        self,
        session: Session,
        user_text: str,
        *,
        speculative: bool = False,
    ) -> PreparedReply:
        """Full prep for speculative reuse (LLM + full TTS buffer)."""
        draft = self.draft_reply(session, user_text)
        t0 = time.perf_counter()
        audio = self.tts.speak_bytes(draft.coach_text)
        tts_ms = int((time.perf_counter() - t0) * 1000)
        return PreparedReply(
            user_text=draft.user_text,
            coach_text=draft.coach_text,
            keywords=draft.keywords,
            audio=audio,
            llm_ms=draft.llm_ms,
            tts_ms=tts_ms,
            prepare_ms=draft.llm_ms + tts_ms,
            speculative=speculative,
        )

    def commit_text(
        self,
        session: Session,
        *,
        user_text: str,
        coach_text: str,
        keywords: list[str],
    ) -> int:
        turn = session.turn + 1
        session.keywords = keywords
        session.add("user", user_text)
        session.add("assistant", coach_text)
        session.turn = turn
        self._persist_async(
            session.session_id,
            turn=turn,
            user_text=user_text,
            coach_text=coach_text,
            keywords=keywords,
        )
        return turn

    def commit_prepared(
        self,
        session: Session,
        prepared: PreparedReply,
        *,
        wait_ms: int = 0,
    ) -> TurnResult:
        """Apply prepared reply to session memory + async mongo."""
        turn = self.commit_text(
            session,
            user_text=prepared.user_text,
            coach_text=prepared.coach_text,
            keywords=prepared.keywords,
        )
        return TurnResult(
            turn=turn,
            user_text=prepared.user_text,
            coach_text=prepared.coach_text,
            keywords=prepared.keywords,
            coach_audio_bytes=prepared.audio,
            used_speculative=prepared.speculative,
            latency={
                "llm_ms": prepared.llm_ms,
                "tts_ms": prepared.tts_ms,
                "ttfb_ms": prepared.ttfb_ms,
                "prepare_ms": prepared.prepare_ms,
                "wait_ms": wait_ms,
                "total_ms": wait_ms,
            },
        )

    def handle_user_text(self, session: Session, user_text: str) -> TurnResult:
        prepared = self.prepare_reply(session, user_text, speculative=False)
        return self.commit_prepared(session, prepared)

    def _persist_async(
        self,
        session_id: str,
        *,
        turn: int,
        user_text: str | None,
        coach_text: str,
        keywords: list[str],
    ) -> None:
        def _run() -> None:
            try:
                if keywords:
                    self.mongo.set_keywords(session_id, keywords)
                if user_text is not None:
                    self.mongo.add_message(
                        session_id, "user", user_text, turn=turn, audio_url=None
                    )
                self.mongo.add_message(
                    session_id,
                    "assistant",
                    coach_text,
                    turn=turn,
                    audio_url=None,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[mongo] persist skip: {exc}")

        threading.Thread(target=_run, daemon=True).start()
