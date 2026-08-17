"""Live coach — prepare (speculative) + commit, with latency timings."""

from __future__ import annotations

import time
from dataclasses import dataclass

from core.session import Session
from core.text_clean import clean_speech_text, clip_spoken_reply
from core.tfidf_engine import TfidfEngine
from wrappers.deepgram_tts import DeepgramTTS
from wrappers.llm import CoachLLM


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
    "do you know",
    "do you remember",
    "know about me",
    "tell me about me",
    "know my name",
)

FILLER_TAIL = frozenset(
    """
    a an the and or but so yeah yes no um uh hmm oh like just very really too
    also is are am was were be been being do does did doing done have has had
    i you he she it we they me my your his her our their this that these those
    in on at of for to from with about by as well now then there here
    still already only even much many some any not dont don't cant can't
    ok okay actually basically mean means i'm you're it's that's thats
    """.split()
)


def needs_fresh_reply(text: str) -> bool:
    """Skip speculative reuse when user asks to clarify — must answer exactly."""
    t = _norm(text)
    if not t:
        return False
    if len(t.split()) <= 2 and t in {"yeah", "yes", "ok", "okay", "i", "huh", "what"}:
        return True
    return any(cue in t for cue in _CLARIFY_CUES)


def _diff_is_filler(a_words: list[str], b_words: list[str]) -> bool:
    i = 0
    while i < len(a_words) and i < len(b_words) and a_words[i] == b_words[i]:
        i += 1
    tail = a_words[i:] + b_words[i:]
    if len(tail) > 6:
        return False
    return all(w in FILLER_TAIL for w in tail)


def transcripts_compatible(partial: str, final: str) -> bool:
    """Reuse speculative prep only when final is basically the same utterance."""
    a, b = _norm(partial), _norm(final)
    if not a or not b:
        return False
    if needs_fresh_reply(final):
        return False
    if a == b:
        return True
    wa, wb = a.split(), b.split()
    if b.startswith(a) or a.startswith(b):
        return _diff_is_filler(wa, wb)
    if len(wa) >= 4 and len(wb) >= 4 and " ".join(wa[:4]) == " ".join(wb[:4]):
        return _diff_is_filler(wa, wb)
    return False


class CoachService:
    def __init__(
        self,
        *,
        tts: DeepgramTTS,
        coach: CoachLLM,
        tfidf: TfidfEngine,
    ) -> None:
        self.tts = tts
        self.coach = coach
        self.tfidf = tfidf

    def open_session(self, session: Session) -> TurnResult:
        starter = "Hi. I'm your English speaking partner. How are you today?"
        session.add("assistant", starter)
        session.turn = 0
        t0 = time.perf_counter()
        audio = self.tts.speak_bytes(clean_speech_text(starter))
        tts_ms = int((time.perf_counter() - t0) * 1000)
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
        t_llm = time.perf_counter()
        history = session.snapshot()[-12:] + [{"role": "user", "content": text}]
        coach_text = (self.coach.reply(history, keywords) or "").strip()
        coach_text = clip_spoken_reply(coach_text, user_text=text)
        if not coach_text:
            coach_text = "Nice. Tell me a bit more."
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
        return turn

    def commit_prepared(
        self,
        session: Session,
        prepared: PreparedReply,
        *,
        wait_ms: int = 0,
    ) -> TurnResult:
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
