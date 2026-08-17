"""App settings — load once, pass around."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.keys import KeyPool
from core.stt_config import DeepgramSTTConfig

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)


def _split_keys(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


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
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    qdrant_vector_size: int
    qdrant_distance: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    embedding_version: str
    tenant_id: str
    users_mongo_db: str
    live_llm_timeout: float
    post_call_llm_timeout: float
    live_llm_max_tokens: int


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
        mongo_uri=(os.getenv("MONGODB_URI") or "").strip(),
        mongo_db=os.getenv("MONGODB_DB", "ai_talk").strip() or "ai_talk",
        qdrant_url=(os.getenv("QDRANT_URL") or "").strip(),
        qdrant_api_key=(os.getenv("QDRANT_API_KEY") or "").strip(),
        qdrant_collection=(
            os.getenv("QDRANT_COLLECTION") or "english_coach_memories"
        ).strip()
        or "english_coach_memories",
        qdrant_vector_size=int(
            os.getenv("EMBEDDING_DIMENSION")
            or os.getenv("QDRANT_VECTOR_SIZE")
            or "384"
        ),
        qdrant_distance=(os.getenv("QDRANT_DISTANCE") or "COSINE").strip().upper()
        or "COSINE",
        embedding_provider=(os.getenv("EMBEDDING_PROVIDER") or "hashing").strip()
        or "hashing",
        embedding_model=(os.getenv("EMBEDDING_MODEL") or "hashing-v1").strip()
        or "hashing-v1",
        embedding_dimension=int(
            os.getenv("EMBEDDING_DIMENSION")
            or os.getenv("QDRANT_VECTOR_SIZE")
            or "384"
        ),
        embedding_version=(os.getenv("EMBEDDING_VERSION") or "v1").strip() or "v1",
        tenant_id=(os.getenv("TENANT_ID") or "talkengly").strip() or "talkengly",
        users_mongo_db=(os.getenv("USERS_MONGO_DB") or "").strip(),
        live_llm_timeout=float(os.getenv("LIVE_LLM_TIMEOUT", "5") or "5"),
        post_call_llm_timeout=float(os.getenv("POST_CALL_LLM_TIMEOUT", "30") or "30"),
        live_llm_max_tokens=int(os.getenv("LIVE_LLM_MAX_TOKENS", "160") or "160"),
    )
