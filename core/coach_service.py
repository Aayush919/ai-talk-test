"""Live coach — prepare (speculative) + commit, with latency timings."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from core.session import Session
from core.text_clean import clean_speech_text
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


_CLARIFY_CUES = (
    "tell me again",
    "say again",
    "repeat",
    "don't get",
    "do not get",
    "dont get",
    "what do you mean",
    "i don't understand",
    "i dont understand",
    "didn't understand",
    "did not understand",
    "confused",
)


def needs_fresh_reply(text: str) -> bool:
    """Skip speculative reuse when user asks to clarify — must answer exactly."""
    t = _norm(text)
    if not t:
        return False
    if len(t.split()) <= 2 and t in {"yeah", "yes", "ok", "okay", "i", "huh", "what"}:
        return True
    return any(cue in t for cue in _CLARIFY_CUES)


def transcripts_compatible(partial: str, final: str) -> bool:
    """Reuse speculative prep only when final is basically the same utterance."""
    a, b = _norm(partial), _norm(final)
    if not a or not b:
        return False
    if needs_fresh_reply(final):
        return False
    if a == b:
        return True
    # Small extension only — big new content must rebuild from full speech
    if b.startswith(a):
        extra = len(b.split()) - len(a.split())
        return extra <= 2
    wa, wb = a.split(), b.split()
    if len(wa) >= 5 and len(wb) >= 5 and " ".join(wa[:5]) == " ".join(wb[:5]):
        return abs(len(wa) - len(wb)) <= 2
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
        audio = self.tts.speak_bytes(clean_speech_text(starter))
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
        # Light grounding — don't force robotic "You said X" echoes
        grounded = (
            f"{text}\n"
            "(Respond naturally to this line. Do not invent facts. "
            "If they asked you something, answer briefly as AI Talk.)"
        )
        history = session.snapshot() + [{"role": "user", "content": grounded}]
        t_llm = time.perf_counter()
        coach_text = (
            self.coach.reply(history, keywords, topic=session.topic) or ""
        ).strip()
        coach_text = clean_speech_text(coach_text)
        if not coach_text:
            coach_text = "Nice - tell me a bit more. What else?"
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
        """Speculative: LLM + TTS of first speak-chunk only (faster ready)."""
        from wrappers.deepgram_tts import split_speak_chunks

        draft = self.draft_reply(session, user_text)
        units = split_speak_chunks(draft.coach_text)
        speak_now = units[0] if units else draft.coach_text
        t0 = time.perf_counter()
        audio = self.tts.speak_bytes(speak_now)
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
