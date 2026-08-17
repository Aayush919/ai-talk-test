"""Persist local wav → Cloudinary."""

from __future__ import annotations

from pathlib import Path

from core.session import Session
from wrappers.cloudinary_store import CloudinaryStore


class MediaPipeline:
    def __init__(self, cloud: CloudinaryStore) -> None:
        self.cloud = cloud

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
        return asset.url
