"""Deepgram TTS — streaming PCM; Aura (/v1) + Flux (/v2) models."""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator

import httpx

from core import call_log
from core.keys import KeyPool
from core.text_clean import safe_print

print = safe_print

DEFAULT_MODEL = (
    os.getenv("DEEPGRAM_TTS_MODEL", "aura-helios-en").strip() or "aura-helios-en"
)


def split_speak_chunks(text: str) -> list[str]:
    """Split reply into short speak units so first audio can start sooner."""
    clean = " ".join((text or "").split())
    if not clean:
        return []
    parts = re.split(r"(?<=[.!?])\s+", clean)
    chunks = [p.strip() for p in parts if p.strip()]
    if len(chunks) >= 2:
        return chunks
    if "," in clean:
        left, _, right = clean.partition(",")
        left, right = left.strip(), right.strip()
        if left and right:
            return [left + ",", right]
    words = clean.split()
    if len(words) > 8:
        return [" ".join(words[:8]), " ".join(words[8:])]
    return [clean]


class DeepgramTTS:
    """Sync WAV helper + async chunked PCM stream (sentence units)."""

    def __init__(self, keys: KeyPool, model: str | None = None) -> None:
        self._keys = keys
        self.model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self._http: httpx.AsyncClient | None = None
        self._speed_ok = True

    @property
    def _is_flux(self) -> bool:
        return self.model.lower().startswith("flux-")

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }

    def _model_url(self, model: str) -> str:
        if model.lower().startswith("flux-"):
            return "https://api.deepgram.com/v2/speak"
        return "https://api.deepgram.com/v1/speak"

    def _model_params(self, model: str, *, use_speed: bool = True) -> dict[str, str]:
        params = {
            "model": model,
            "encoding": "linear16",
            "sample_rate": "16000",
            "container": "none",
        }
        speed = (os.getenv("DEEPGRAM_TTS_SPEED") or "0.90").strip()
        if (
            use_speed
            and self._speed_ok
            and speed
            and not model.lower().startswith("flux-")
        ):
            params["speed"] = speed
        return params

    def _candidates(self) -> list[str]:
        models = [self.model]
        if self._is_flux and "aura-helios-en" not in models:
            models.append("aura-helios-en")
        return models

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
        return self._http

    async def aclose(self) -> None:
        if self._http is not None and not self._http.is_closed:
            await self._http.aclose()
        self._http = None

    def speak_bytes(self, text: str) -> bytes:
        """Blocking full WAV (session opener / speculative)."""
        clean = (text or "").strip()
        if not clean:
            raise RuntimeError("TTS skipped: empty text")

        def _once(api_key: str) -> bytes:
            last_err: Exception | None = None
            for model in self._candidates():
                for use_speed in (True, False):
                    if use_speed and not self._speed_ok:
                        continue
                    try:
                        with httpx.Client(timeout=30.0) as client:
                            with client.stream(
                                "POST",
                                self._model_url(model),
                                params=self._model_params(model, use_speed=use_speed),
                                headers=self._headers(api_key),
                                json={"text": clean},
                            ) as resp:
                                if resp.status_code >= 400:
                                    body = resp.read()
                                    err = RuntimeError(
                                        f"Deepgram TTS {resp.status_code}: {body[:200]!r}"
                                    )
                                    if use_speed and resp.status_code == 400:
                                        self._speed_ok = False
                                        print("[tts] speed rejected — retrying without speed")
                                        last_err = err
                                        continue
                                    raise err
                                data = b"".join(resp.iter_bytes())
                        if not data:
                            raise RuntimeError("Deepgram TTS returned empty audio")
                        if model != self.model:
                            print(f"[tts] fallback model={model}")
                            self.model = model
                        return self.pcm_to_wav(data)
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        print(f"[tts] model={model} fail: {exc}")
                        call_log.error("TTS", str(exc), extra={"model": model})
                        break
            raise RuntimeError(str(last_err) if last_err else "TTS failed")

        return self._keys.run(_once)

    async def stream_pcm(self, text: str) -> AsyncIterator[bytes]:
        """Yield raw PCM for one text unit."""
        clean = (text or "").strip()
        if not clean:
            raise RuntimeError("TTS skipped: empty text")

        last_err: Exception | None = None
        for model in self._candidates():
            for attempt in (1, 2):
                try:
                    api_key = self._keys.pick()
                    client = await self._client()
                    async with client.stream(
                        "POST",
                        self._model_url(model),
                        params=self._model_params(model),
                        headers=self._headers(api_key),
                        json={"text": clean},
                    ) as resp:
                        if resp.status_code >= 400:
                            body = await resp.aread()
                            if self._speed_ok and resp.status_code == 400:
                                self._speed_ok = False
                                print("[tts] stream speed rejected — retrying without speed")
                                continue
                            raise RuntimeError(
                                f"Deepgram TTS {resp.status_code}: {body[:200]!r}"
                            )
                        if model != self.model:
                            print(f"[tts] stream fallback model={model}")
                            self.model = model
                        async for chunk in resp.aiter_bytes(chunk_size=1024):
                            if chunk:
                                yield chunk
                    return
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    transient = "disconnected" in str(exc).lower()
                    print(f"[tts] stream model={model} fail: {exc}")
                    call_log.error("TTS", f"stream fail: {exc}", extra={"model": model})
                    if transient and attempt == 1:
                        continue
                    break
        raise RuntimeError(str(last_err) if last_err else "TTS stream failed")

    async def stream_pcm_chunked(self, text: str) -> AsyncIterator[bytes]:
        """First sentence/clause first — lower time-to-hear, then continue."""
        units = split_speak_chunks(text)
        if not units:
            return
        for i, unit in enumerate(units):
            print(f"[tts] chunk {i + 1}/{len(units)}: {unit!r}")
            async for pcm in self.stream_pcm(unit):
                yield pcm

    def pcm_to_wav(self, pcm: bytes, sample_rate: int = 16000) -> bytes:
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
