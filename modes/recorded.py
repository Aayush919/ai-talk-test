"""Recorded mode — mic clip → local wav → Cloudinary → Deepgram STT."""

from __future__ import annotations

from core.config import Settings
from core.session import Session
from wrappers.audio_io import record_wav
from wrappers.deepgram_stt import DeepgramSTT
from wrappers.media_pipeline import MediaPipeline


class RecordedMode:
    name = "recorded"

    def __init__(
        self,
        settings: Settings,
        stt: DeepgramSTT,
        media: MediaPipeline,
    ) -> None:
        self._settings = settings
        self._stt = stt
        self._media = media

    def listen(self, session: Session, turn: int) -> str:
        assert session.audio_dir is not None
        path = session.audio_dir / f"turn_{turn:03d}_user.wav"
        print(f"[recorded] Speak now ({self._settings.record_seconds}s)...")
        record_wav(
            path,
            seconds=self._settings.record_seconds,
            sample_rate=self._settings.sample_rate,
        )
        session.last_user_audio_url = self._media.save_clip(
            session,
            turn=turn,
            role="user",
            path=path,
        )
        text = self._stt.transcribe_file(path)
        print(f"[you] {text}")
        print(f"[cloud] {session.last_user_audio_url}")
        return text
