"""Deepgram TTS — streaming PCM for low time-to-first-byte."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from core.keys import KeyPool

DEFAULT_MODEL = os.getenv("DEEPGRAM_TTS_MODEL", "aura-helios-en").strip() or "aura-helios-en"
DEFAULT_SPEED = (os.getenv("DEEPGRAM_TTS_SPEED", "1.2").strip() or "1.2")


class DeepgramTTS:
    """
    Sync full-audio helper (session opener / speculative buffer)
    + async stream for live turns (send chunks as soon as they arrive).
    """

    def __init__(self, keys: KeyPool, model: str | None = None) -> None:
        self._keys = keys
        self.model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.speed = DEFAULT_SPEED
        self._http: httpx.AsyncClient | None = None

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }

    def _params(self) -> dict[str, str]:
        return {
            "model": self.model,
            "encoding": "linear16",
            "sample_rate": "16000",
            "container": "none",
            "speed": self.speed,
        }

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    def speak_bytes(self, text: str) -> bytes:
        """Blocking full PCM (used for speculative prep / opener)."""
        clean = (text or "").strip()
        if not clean:
            raise RuntimeError("TTS skipped: empty text")

        def _once(api_key: str) -> bytes:
            with httpx.Client(timeout=30.0) as client:
                with client.stream(
                    "POST",
                    "https://api.deepgram.com/v1/speak",
                    params=self._params(),
                    headers=self._headers(api_key),
                    json={"text": clean},
                ) as resp:
                    if resp.status_code >= 400:
                        body = resp.read()
                        raise RuntimeError(
                            f"Deepgram TTS {resp.status_code}: {body[:200]!r}"
                        )
                    chunks = list(resp.iter_bytes())
            data = b"".join(chunks)
            if not data:
                raise RuntimeError("Deepgram TTS returned empty audio")
            return self.pcm_to_wav(data)

        return self._keys.run(_once)

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw PCM chunks as soon as Deepgram produces them."""
        clean = (text or "").strip()
        if not clean:
            raise RuntimeError("TTS skipped: empty text")

        api_key = self._keys.pick()
        client = await self._client()
        async with client.stream(
            "POST",
            "https://api.deepgram.com/v1/speak",
            params=self._params(),
            headers=self._headers(api_key),
            json={"text": clean},
        ) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise RuntimeError(f"Deepgram TTS {resp.status_code}: {body[:200]!r}")
            async for chunk in resp.aiter_bytes(chunk_size=2048):
                if chunk:
                    yield chunk

    def pcm_to_wav(self, pcm: bytes, sample_rate: int = 16000) -> bytes:
        """Wrap PCM for browsers that only play WAV blobs (fallback)."""
        import struct

        channels = 1
        bits = 16
        data_size = len(pcm)
        byte_rate = sample_rate * channels * bits // 8
        block_align = channels * bits // 8
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits,
            b"data",
            data_size,
        )
        return header + pcm
