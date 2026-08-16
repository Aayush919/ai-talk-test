"""MongoDB wrapper — sessions, topics seed, transcripts + Cloudinary URLs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection

from core.topics import load_topics


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MongoStore:
    def __init__(self, uri: str, db_name: str = "ai_talk") -> None:
        # Fail fast if Atlas is slow — don't freeze page reloads / startup
        self._client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=10000,
        )
        self._db = self._client[db_name]
        self.sessions: Collection = self._db["sessions"]
        self.topics: Collection = self._db["topics"]
        self.meta: Collection = self._db["meta"]
        self.learner_profiles: Collection = self._db["learner_profiles"]
        self.learner_vectors: Collection = self._db["learner_vectors"]
        try:
            self.sessions.create_index("session_id", unique=True)
            self.topics.create_index("id", unique=True)
        except Exception:
            # Index creation can wait; don't block app boot
            pass

    def ensure_memory_indexes(self) -> None:
        try:
            self.learner_profiles.create_index("learner_id", unique=True)
            self.learner_vectors.create_index("learner_id")
            self.learner_vectors.create_index(
                [("learner_id", 1), ("doc_id", 1)], unique=True
            )
        except Exception:
            pass

    def seed_topics(self, *, force: bool = False) -> int:
        """Seed topics once. Skip if already done (unless force=True)."""
        flag = self.meta.find_one({"_id": "topics_seed"})
        if flag and not force:
            return 0
        if not force and self.topics.estimated_document_count() > 0:
            self.meta.update_one(
                {"_id": "topics_seed"},
                {"$set": {"done": True, "at": _utc_now()}},
                upsert=True,
            )
            return 0

        count = 0
        for topic in load_topics():
            self.topics.update_one(
                {"id": topic.id},
                {
                    "$set": {
                        **topic.as_dict(),
                        "updated_at": _utc_now(),
                    },
                    "$setOnInsert": {"created_at": _utc_now()},
                },
                upsert=True,
            )
            count += 1

        self.meta.update_one(
            {"_id": "topics_seed"},
            {"$set": {"done": True, "count": count, "at": _utc_now()}},
            upsert=True,
        )
        return count

    def list_topics(self) -> list[dict[str, Any]]:
        rows = list(self.topics.find({}, {"_id": 0}).sort("title", 1))
        return rows or [t.as_dict() for t in load_topics()]

    def create_session(
        self,
        session_id: str,
        mode: str,
        *,
        topic_id: str | None = None,
        topic_title: str | None = None,
    ) -> None:
        now = _utc_now()
        doc: dict[str, Any] = {
            "session_id": session_id,
            "mode": mode,
            "topic_id": topic_id,
            "topic_title": topic_title,
            "keywords": [],
            "messages": [],
            "audios": [],
            "created_at": now,
            "updated_at": now,
        }
        self.sessions.insert_one(doc)

    def set_keywords(self, session_id: str, keywords: list[str]) -> None:
        self.sessions.update_one(
            {"session_id": session_id},
            {"$set": {"keywords": keywords, "updated_at": _utc_now()}},
        )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        turn: int,
        audio_url: str | None = None,
    ) -> None:
        msg: dict[str, Any] = {
            "role": role,
            "content": content,
            "turn": turn,
            "at": _utc_now(),
        }
        if audio_url:
            msg["audio_url"] = audio_url
        self.sessions.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": msg},
                "$set": {"updated_at": _utc_now()},
            },
        )

    def add_audio(
        self,
        session_id: str,
        *,
        turn: int,
        role: str,
        url: str,
        public_id: str,
    ) -> None:
        self.sessions.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "audios": {
                        "turn": turn,
                        "role": role,
                        "url": url,
                        "public_id": public_id,
                        "at": _utc_now(),
                    }
                },
                "$set": {"updated_at": _utc_now()},
            },
        )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self.sessions.find_one({"session_id": session_id}, {"_id": 0})

    def get_learner_profile(self, learner_id: str) -> dict[str, Any] | None:
        return self.learner_profiles.find_one({"learner_id": learner_id}, {"_id": 0})

    def upsert_learner_profile(self, learner_id: str, facts: dict[str, str]) -> None:
        if not learner_id or not facts:
            return
        self.learner_profiles.update_one(
            {"learner_id": learner_id},
            {
                "$set": {"facts": facts, "updated_at": _utc_now()},
                "$setOnInsert": {"learner_id": learner_id, "created_at": _utc_now()},
            },
            upsert=True,
        )

    def upsert_learner_vector(
        self,
        *,
        learner_id: str,
        doc_id: str,
        text: str,
        kind: str,
        vector: list[float],
        extra: dict[str, Any],
    ) -> None:
        self.learner_vectors.update_one(
            {"learner_id": learner_id, "doc_id": doc_id},
            {
                "$set": {
                    "text": text,
                    "kind": kind,
                    "vector": vector,
                    "extra": extra,
                    "updated_at": _utc_now(),
                },
                "$setOnInsert": {"created_at": _utc_now()},
            },
            upsert=True,
        )

    def query_learner_vectors(
        self, learner_id: str, query_vec: list[float], top_k: int = 5
    ) -> list[str]:
        """Per-user cosine — O(this learner's docs), not O(all users)."""
        import numpy as np

        rows = list(
            self.learner_vectors.find(
                {"learner_id": learner_id},
                {"_id": 0, "text": 1, "vector": 1},
            ).limit(80)
        )
        if not rows:
            return []
        q = np.array(query_vec, dtype=float)
        scored: list[tuple[float, str]] = []
        qn = float(np.linalg.norm(q)) or 1e-9
        for row in rows:
            vec = row.get("vector") or []
            if not vec:
                continue
            v = np.array(vec, dtype=float)
            denom = qn * (float(np.linalg.norm(v)) or 1e-9)
            scored.append((float(np.dot(q, v) / denom), str(row.get("text") or "")))
        scored.sort(reverse=True)
        return [t for _, t in scored[:top_k] if t]
