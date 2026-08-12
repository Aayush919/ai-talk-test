"""Mic capture helpers — recorded clip for the recorded mode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def record_wav(path: Path, seconds: float, sample_rate: int) -> Path:
    frames = int(seconds * sample_rate)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    sf.write(str(path), audio, sample_rate)
    return path


def play_wav(path: Path) -> None:
    data, rate = sf.read(str(path), dtype="float32")
    sd.play(data, rate)
    sd.wait()


def play_audio_bytes(raw: bytes, sample_rate: int = 16000) -> None:
    """Best-effort PCM/WAV play; prefers soundfile when container is wav."""
    import io

    try:
        data, rate = sf.read(io.BytesIO(raw), dtype="float32")
        sd.play(data, rate)
        sd.wait()
        return
    except Exception:
        pass

    # Fallback: treat as mono float32 PCM
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size == 0:
        return
    sd.play(arr, sample_rate)
    sd.wait()
