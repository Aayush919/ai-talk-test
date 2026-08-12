"""Deepgram STT wrapper — nova-3, en-IN chain, endpointing for live."""

from __future__ import annotations

from pathlib import Path

from deepgram import DeepgramClient

from core.keys import KeyPool
from core.stt_config import DeepgramSTTConfig


def _transcript_from(response: object) -> str:
    results = getattr(response, "results", None)
    channels = getattr(results, "channels", None) if results else None
    if not channels:
        return ""
    alts = getattr(channels[0], "alternatives", None) or []
    if not alts:
        return ""
    return (getattr(alts[0], "transcript", None) or "").strip()


class DeepgramSTT:
    def __init__(self, keys: KeyPool, config: DeepgramSTTConfig) -> None:
        self._keys = keys
        self.config = config

    def transcribe_file(self, audio_path: Path) -> str:
        return self.transcribe_bytes(audio_path.read_bytes())

    def transcribe_bytes(self, audio: bytes) -> str:
        # Language fallback chain: en-IN → en → en-US (first non-empty wins)
        for language in self.config.language_chain:
            text = self._transcribe_once(audio, language=language)
            if text:
                return text
        return ""

    def _transcribe_once(self, audio: bytes, *, language: str) -> str:
        def _once(api_key: str) -> str:
            client = DeepgramClient(api_key=api_key)
            response = client.listen.v1.media.transcribe_file(
                request=audio,
                model=self.config.model,
                language=language,
                smart_format=True,
                punctuate=True,
                request_options={"timeout": 120.0},
            )
            return _transcript_from(response)

        return self._keys.run(_once)

    def live_connect_kwargs(self) -> dict:
        """Options for realtime websocket listen (endpointing lives here)."""
        return {
            "model": self.config.model,
            "language": self.config.language,
            "smart_format": True,
            "punctuate": True,
            "endpointing": self.config.endpointing_ms,
        }
