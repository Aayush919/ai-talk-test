"""Persist local wav → Cloudinary, then register URL in Mongo."""

from __future__ import annotations

from pathlib import Path

from core.session import Session
from wrappers.cloudinary_store import CloudinaryStore
from wrappers.mongo_store import MongoStore


class MediaPipeline:
    def __init__(self, cloud: CloudinaryStore, mongo: MongoStore) -> None:
        self.cloud = cloud
        self.mongo = mongo

    def save_clip(
        self,
        session: Session,
        *,
        turn: int,
        role: str,
        path: Path,
    ) -> str:
        asset = self.cloud.upload_audio(
            path,
            session_id=session.session_id,
            turn=turn,
            role=role,
        )
        self.mongo.add_audio(
            session.session_id,
            turn=turn,
            role=role,
            url=asset.url,
            public_id=asset.public_id,
        )
        return asset.url
