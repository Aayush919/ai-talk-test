"""Live coach — prepare (speculative) + commit, with latency timings."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from core import call_log
from core.learn.teach import teach_line
from core.memory.extract import (
    FILLER_TAIL,
    is_fragment,
    quoted_target,
    strip_unfaithful_fix,
)
from core.memory.bank import SCENE_TOPICS
from core.memory.vector import VectorMemory
from core.session import Session
from core.text_clean import clean_speech_text, clip_spoken_reply
from core.tfidf_engine import TfidfEngine
from wrappers.deepgram_tts import DeepgramTTS
from wrappers.llm import CoachLLM
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
    "do you know",
    "know about me",
    "tell me about me",
    "know my name",
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
    """True only if the words the final adds/drops carry no meaning.

    The answer itself often arrives in the last word ("...is Lakeview"), so a
    draft built without it must never be reused.
    """
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
        mongo: MongoStore,
        vectors: VectorMemory | None = None,
    ) -> None:
        self.tts = tts
        self.coach = coach
        self.tfidf = tfidf
        self.mongo = mongo
        self.vectors = vectors

    def open_session(self, session: Session) -> TurnResult:
        self._hydrate_semantic(session)
        mem = session.memory
        if mem and not (session.topic and session.topic.id in SCENE_TOPICS):
            mem.call_warm = True
        starter = self._opening_line(session)
        session.add("assistant", starter)
        if mem:
            mem.note_coach_reply(starter)
        session.turn = 0
        self._hydrate_vectors_async(session)
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

    def _opening_line(self, session: Session) -> str:
        topic = session.topic
        mem = session.memory
        if topic and topic.id in SCENE_TOPICS:
            return topic.starter
        name = mem.display_name() if mem else ""
        if name:
            return f"Hey {name}, good to have you back. How are you today?"
        return "Hi. I'm your English speaking partner. How are you today?"

    def draft_reply(self, session: Session, user_text: str) -> DraftReply:
        """LLM only — TTS streams separately for lower time-to-first-audio."""
        text = (user_text or "").strip()
        history_texts = [m["content"] for m in session.messages]
        keywords = self.tfidf.extract(history_texts, text)
        mem = session.memory
        pack = mem.prompt_pack() if mem else ""
        t_llm = time.perf_counter()
        if mem:
            recap = mem.spoken_recall(text)
            if recap:
                return DraftReply(
                    user_text=text,
                    coach_text=recap,
                    keywords=keywords,
                    llm_ms=int((time.perf_counter() - t_llm) * 1000),
                )
            model = mem.model_sentence()
            if model and not mem.call_warm:
                coach_text = teach_line(text, model)
                return DraftReply(
                    user_text=text,
                    coach_text=coach_text,
                    keywords=keywords,
                    llm_ms=int((time.perf_counter() - t_llm) * 1000),
                )
        hint = (
            "Stay on the current speaking level. Reply to this line. "
            "If they went off-topic, follow then return. "
            "Do not correct grammar unless the memory card says CORRECT yes."
        )
        if mem:
            if mem.call_warm:
                about = mem.topic_about()
                seed = (
                    mem._follow_up_question()
                    if mem._lesson_complete()
                    else mem._seed_question()
                )
                if not seed or mem._q_asked(seed):
                    seed = mem.fresh_question()
                if mem.display_name():
                    hint = (
                        "Greet back briefly. Do not correct grammar. "
                        f"Then continue from last time ({about}). Ask: {seed or about}. "
                        "Never repeat a question you already asked."
                    )
                else:
                    hint = (
                        "Greet back briefly. Do not correct grammar. "
                        "Then start introducing themselves. Ask: What is your name?"
                    )
            elif is_fragment(text):
                hint = (
                    "Their sentence is incomplete. Ask them to say the whole sentence. "
                    "Do not copy the broken fragment. Do not change topic."
                )
            elif mem.filled_now:
                follow = mem._follow_up_question() or mem.fresh_question()
                hint = (
                    f"They just answered {mem.topic_about()}. React to that. "
                    f"Ask one NEW follow-up on THIS topic: {follow}. "
                    "Never repeat an old question. Do not jump to the next level."
                )
            elif mem.hold_lesson:
                nxt = mem.next_about()
                hint = (
                    "Reply to this line first. Stay on this topic if they are still talking. "
                    + (f"Only if that thread is done, you may move to {nxt}. " if nxt else "")
                    + f"Ask a NEW question: {mem.fresh_question()}. Never repeat an old question."
                )
            else:
                model = mem.model_sentence()
                if model:
                    hint = (
                        f'Small note: say it like this: "{model}" '
                        "Ask them to try it. Do not treat their name as an activity."
                    )
                elif mem.intel.correct_now:
                    hint = (
                        'Teach: You said "...". We don\'t say it like that. '
                        'Say it like this: "..." Try it. No new question this turn.'
                    )
                repeat = mem.repeat_peek(text)
                if repeat == "matched":
                    hint = (
                        "They repeated the corrected sentence. Brief praise, "
                        f"then this NEW question: {mem.fresh_question()}. Never repeat an old question."
                    )
                elif repeat == "skipped":
                    hint = (
                        "They moved on without repeating. Do not explain the fix again. "
                        f"Ask this NEW question: {mem.fresh_question()}."
                    )
        grounded = f"{text}\n({hint})"
        # STM: last 6 turns already trimmed in Groq; keep snapshot small here too
        history = session.snapshot()[-12:] + [{"role": "user", "content": grounded}]
        coach_text = (
            self.coach.reply(
                history,
                keywords,
                topic=session.topic,
                memory_block=pack,
            )
            or ""
        ).strip()
        coach_text = clip_spoken_reply(coach_text, user_text=text)
        coach_text = strip_unfaithful_fix(coach_text, text)
        if mem:
            coach_text = mem.dedupe_reply(coach_text)
        if not coach_text:
            coach_text = mem.fresh_question() if mem else "Nice. Tell me a bit more."
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
        if session.memory:
            mem = session.memory
            mem.observe_user(user_text, turn)
            mem.repeat_clear()
            mem.note_coach_reply(coach_text)
            target = quoted_target(coach_text)
            if target and not mem.already_corrected(target):
                mem.start_repeat(target)
                if mem.intel.correct_now:
                    mem.intel.note_correction(mem.intel.focus)
            call_log.info(
                "MEMORY",
                "turn consolidated",
                session_id=session.session_id,
                extra={
                    "learner": session.memory.learner_id,
                    "turn": turn,
                    "user": (user_text or "")[:80],
                    "coach": (coach_text or "")[:80],
                    "semantic": session.memory.fact_card(),
                    "goal": session.memory.next_goal(),
                    "lesson": session.memory.lesson,
                    "intel": session.memory.intel.compact_line(),
                    "episodic": session.memory.episode_line(),
                },
            )
            self._persist_memory_async(session)
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
                call_log.warn("MONGO", f"persist skip: {exc}", session_id=session_id)

        threading.Thread(target=_run, daemon=True).start()

    def _hydrate_semantic(self, session: Session) -> None:
        mem = session.memory
        if not mem:
            return
        try:
            doc = self.mongo.get_learner_profile(mem.learner_id)
            facts = (doc or {}).get("facts") or {}
            if isinstance(facts, dict):
                mem.hydrate_semantic({k: str(v) for k, v in facts.items() if v})
            call_log.info(
                "MEMORY",
                "hydrated",
                session_id=session.session_id,
                extra={
                    "learner": mem.learner_id,
                    "semantic": mem.fact_card(),
                    "phase": mem.phase,
                },
            )
        except Exception as exc:  # noqa: BLE001
            call_log.warn(
                "MEMORY",
                f"hydrate skip: {exc}",
                session_id=session.session_id,
            )

    def _hydrate_vectors_async(self, session: Session) -> None:
        mem = session.memory
        if not mem or not self.vectors:
            return

        def _run() -> None:
            try:
                q = f"{session.topic.title if session.topic else 'talk'} {mem.fact_card()}"
                mem.ltm_snippets = self.vectors.hydrate(mem.learner_id, q)
                call_log.info(
                    "MEMORY",
                    "vectors ready",
                    session_id=session.session_id,
                    extra={
                        "learner": mem.learner_id,
                        "ltm": "; ".join(mem.ltm_snippets[:3]) or "(none)",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                call_log.warn(
                    "MEMORY",
                    f"vector hydrate skip: {exc}",
                    session_id=session.session_id,
                )

        threading.Thread(target=_run, daemon=True).start()

    def _persist_memory_async(self, session: Session) -> None:
        mem = session.memory
        if not mem:
            return
        mem.background_tick()
        facts = mem.persistable_facts()

        def _run() -> None:
            try:
                self.mongo.upsert_learner_profile(mem.learner_id, facts)
                call_log.info(
                    "MEMORY",
                    "profile saved",
                    session_id=session.session_id,
                    extra={"learner": mem.learner_id, "facts": mem.fact_card()},
                )
            except Exception as exc:  # noqa: BLE001
                call_log.warn(
                    "MEMORY",
                    f"profile skip: {exc}",
                    session_id=session.session_id,
                )

        threading.Thread(target=_run, daemon=True).start()
        if self.vectors and facts:
            card = mem.fact_card()
            self.vectors.upsert_async(
                learner_id=mem.learner_id,
                doc_id=f"{mem.learner_id}-semantic",
                text=card,
                kind="semantic",
                extra={"topic": mem.topic_id, "session_id": session.session_id},
            )
            if mem.episodes:
                self.vectors.upsert_async(
                    learner_id=mem.learner_id,
                    doc_id=f"{session.session_id}-ep-{session.turn}",
                    text=mem.episode_line(),
                    kind="episodic",
                    extra={"session_id": session.session_id, "turn": session.turn},
                )
