"""Main conversation loop — listen → TF-IDF → Groq → TTS."""

from __future__ import annotations

from pathlib import Path

from core.session import Session
from core.tfidf_engine import TfidfEngine
from modes.base import TalkMode
from wrappers.audio_io import play_wav
from wrappers.deepgram_tts import DeepgramTTS
from wrappers.groq_llm import GroqCoach
from wrappers.media_pipeline import MediaPipeline

QUIT_WORDS = frozenset({"quit", "exit", "stop", "bye", "goodbye"})


class TalkLoop:
    def __init__(
        self,
        mode: TalkMode,
        coach: GroqCoach,
        tts: DeepgramTTS,
        tfidf: TfidfEngine,
        media: MediaPipeline,
    ) -> None:
        self.mode = mode
        self.coach = coach
        self.tts = tts
        self.tfidf = tfidf
        self.media = media

    def run(self, session: Session) -> None:
        print(f"AI Talk started | mode={session.mode} | session={session.session_id}")
        print("Say 'quit' to stop.\n")

        opener = (
            "Hi! I'm your English talk coach. "
            "What would you like to practice today?"
        )
        session.add("assistant", opener)
        self._speak(session, 0, opener)

        turn = 1
        while True:
            user_text = self.mode.listen(session, turn).strip()
            if not user_text:
                print("[skip] No speech detected — try again.")
                continue

            if user_text.lower().rstrip(".!") in QUIT_WORDS:
                print("Coach: Great practice — see you next time!")
                break

            history_texts = [m["content"] for m in session.messages]
            keywords = self.tfidf.extract(history_texts, user_text)
            session.keywords = keywords

            session.add("user", user_text)
            reply = self.coach.reply(session.snapshot(), keywords)
            session.add("assistant", reply)
            print(f"[coach] {reply}")
            if keywords:
                print(f"[tf-idf] {', '.join(keywords)}")

            self._speak(session, turn, reply)
            turn += 1

    def _speak(self, session: Session, turn: int, text: str) -> str:
        audio_dir = getattr(session, "audio_dir", None)
        assert audio_dir is not None
        out = Path(audio_dir) / f"turn_{turn:03d}_coach.wav"
        path: Path = self.tts.speak_to_file(text, out)
        play_wav(path)
        url = self.media.save_clip(session, turn=turn, role="coach", path=path)
        print(f"[cloud] {url}")
        return url
