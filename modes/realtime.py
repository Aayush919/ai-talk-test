"""
Realtime mode — short rolling capture that feels live.
Same Cloudinary + STT path as recorded; tighter window.
"""

from __future__ import annotations

from core.config import Settings
from core.session import Session
from wrappers.audio_io import record_wav
from wrappers.deepgram_stt import DeepgramSTT
from wrappers.media_pipeline import MediaPipeline


class RealtimeMode:
    name = "realtime"

    def __init__(
        self,
        settings: Settings,
        stt: DeepgramSTT,
        media: MediaPipeline,
    ) -> None:
        self._settings = settings
        self._stt = stt
        self._media = media
        self._window = min(settings.record_seconds, 3.0)

    def listen(self, session: Session, turn: int) -> str:
        assert session.audio_dir is not None
        path = session.audio_dir / f"turn_{turn:03d}_live.wav"
        print(f"[realtime] Listening ({self._window}s window)...")
        record_wav(
            path,
            seconds=self._window,
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
