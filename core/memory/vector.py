"""Vector memory — query only off the live STT/LLM/TTS path.

Per-learner fetch (indexed) scales to 100k users: never scan the whole corpus.
Pinecone if PINECONE_API_KEY is set; else Mongo + hashed embeddings (same API).
"""

from __future__ import annotations

import os
import threading
from typing import Any

from sklearn.feature_extraction.text import HashingVectorizer

from core import call_log
from wrappers.mongo_store import MongoStore

EMBED_DIM = 384
_hasher = HashingVectorizer(
    n_features=EMBED_DIM,
    alternate_sign=False,
    ngram_range=(1, 2),
    norm="l2",
)


def embed_text(text: str) -> list[float]:
    raw = (text or "").strip() or "empty"
    vec = _hasher.transform([raw]).toarray()[0]
    return vec.astype(float).tolist()


class VectorMemory:
    def __init__(self, mongo: MongoStore) -> None:
        self._mongo = mongo
        self._pinecone = None
        self._index = None
        key = (os.getenv("PINECONE_API_KEY") or "").strip()
        index_name = (os.getenv("PINECONE_INDEX") or "ai-talk-memory").strip()
        if key:
            try:
                from pinecone import Pinecone

                pc = Pinecone(api_key=key)
                self._pinecone = pc
                self._index = pc.Index(index_name)
                call_log.info("VECTOR", f"pinecone ready index={index_name}")
            except Exception as exc:  # noqa: BLE001
                call_log.warn("VECTOR", f"pinecone off, using mongo: {exc}")
                self._pinecone = None
                self._index = None
        else:
            call_log.info("VECTOR", "mongo per-learner vectors (no PINECONE_API_KEY)")
        mongo.ensure_memory_indexes()

    def hydrate(self, learner_id: str, query: str) -> list[str]:
        """Top memories for this learner only — call-start / background."""
        try:
            if self._index is not None:
                res = self._index.query(
                    vector=embed_text(query),
                    top_k=5,
                    namespace=learner_id[:64],
                    include_metadata=True,
                )
                matches = getattr(res, "matches", None) or res.get("matches") or []
                out = []
                for m in matches:
                    meta = getattr(m, "metadata", None) or m.get("metadata") or {}
                    text = meta.get("text")
                    if text:
                        out.append(str(text)[:180])
                return out
            return self._mongo.query_learner_vectors(learner_id, embed_text(query), top_k=5)
        except Exception as exc:  # noqa: BLE001
            call_log.warn("MEMORY", f"hydrate skip: {exc}")
            return []

    def upsert_async(
        self,
        *,
        learner_id: str,
        doc_id: str,
        text: str,
        kind: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        def _run() -> None:
            try:
                vec = embed_text(text)
                meta = {"text": text[:400], "kind": kind, **(extra or {})}
                if self._index is not None:
                    self._index.upsert(
                        vectors=[{"id": doc_id[:64], "values": vec, "metadata": meta}],
                        namespace=learner_id[:64],
                    )
                self._mongo.upsert_learner_vector(
                    learner_id=learner_id,
                    doc_id=doc_id,
                    text=text,
                    kind=kind,
                    vector=vec,
                    extra=extra or {},
                )
                call_log.info(
                    "VECTOR",
                    "upsert ok",
                    extra={
                        "learner": learner_id,
                        "kind": kind,
                        "backend": "pinecone" if self._index is not None else "mongo",
                        "text": text[:80],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                call_log.warn("VECTOR", f"upsert skip: {exc}")

        threading.Thread(target=_run, daemon=True).start()
