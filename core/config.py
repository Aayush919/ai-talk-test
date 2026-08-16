"""App settings — load once, pass around."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from core.keys import KeyPool
from core.stt_config import DeepgramSTTConfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)


def _split_keys(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def _require(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} missing in .env")
    return value


@dataclass(frozen=True)
class Settings:
    groq_keys: KeyPool | None
    deepgram_stt_keys: KeyPool
    deepgram_tts_keys: KeyPool
    groq_model: str
    llm_provider: str
    sarvam_keys: KeyPool | None
    sarvam_model: str
    deepgram_stt: DeepgramSTTConfig
    sample_rate: int
    mongo_uri: str
    mongo_db: str


def load_settings() -> Settings:
    groq = _split_keys(os.getenv("GROQ_API_KEYS", ""))
    deepgram = _split_keys(os.getenv("DEEPGRAM_API_KEYS", ""))
    sarvam = _split_keys(
        os.getenv("SARVAM_API_KEYS", "") or os.getenv("SARVAM_API_KEY", "")
    )
    provider = (os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    if provider not in {"groq", "sarvam"}:
        provider = "groq"

    if not deepgram:
        raise RuntimeError("DEEPGRAM_API_KEYS missing in .env")
    if provider == "sarvam" and not sarvam:
        raise RuntimeError(
            "LLM_PROVIDER=sarvam but SARVAM_API_KEYS missing. "
            "Get a key from https://dashboard.sarvam.ai"
        )
    if provider == "groq" and not groq:
        raise RuntimeError("GROQ_API_KEYS missing in .env")

    return Settings(
        groq_keys=KeyPool(groq) if groq else None,
        deepgram_stt_keys=KeyPool(deepgram),
        deepgram_tts_keys=KeyPool(deepgram),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        llm_provider=provider,
        sarvam_keys=KeyPool(sarvam) if sarvam else None,
        sarvam_model=(
            os.getenv("SARVAM_MODEL", "sarvam-105b-conversations").strip()
            or "sarvam-105b-conversations"
        ),
        deepgram_stt=DeepgramSTTConfig(
            model=os.getenv("DEEPGRAM_STT_MODEL", "nova-3").strip() or "nova-3",
            language=os.getenv("DEEPGRAM_STT_LANGUAGE", "en-IN").strip()
            or "en-IN",
            language_fallbacks=tuple(
                _split_keys(
                    os.getenv("DEEPGRAM_STT_LANGUAGE_FALLBACKS", "en,en-US")
                )
            ),
            endpointing_ms=int(os.getenv("DEEPGRAM_STT_ENDPOINTING_MS", "200")),
        ),
        sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
        mongo_uri=_require("MONGODB_URI"),
        mongo_db=os.getenv("MONGODB_DB", "ai_talk").strip() or "ai_talk",
    )
