"""Cloudinary wrapper — audio files go to cloud, not DB blobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cloudinary
import cloudinary.uploader


@dataclass(frozen=True)
class CloudAsset:
    url: str
    public_id: str


class CloudinaryStore:
    def __init__(
        self,
        cloud_name: str,
        api_key: str,
        api_secret: str,
        folder: str = "ai-talk",
        cdn_subdomain: bool = True,
    ) -> None:
        cloudinary.config(
            cloud_name=cloud_name,
            api_key=api_key,
            api_secret=api_secret,
            secure=True,
            cdn_subdomain=cdn_subdomain,
        )
        self.folder = folder.strip("/")

    def upload_audio(
        self,
        path: Path,
        *,
        session_id: str,
        turn: int,
        role: str,
    ) -> CloudAsset:
        public_id = f"{self.folder}/{session_id}/turn_{turn:03d}_{role}"
        ext = path.suffix.lstrip(".").lower() or "wav"
        options = {
            "resource_type": "video",
            "public_id": public_id,
            "overwrite": True,
        }
        # Keep coach wav as wav; browser mic often sends webm/opus
        if ext in {"wav", "mp3", "ogg", "mp4", "m4a"}:
            options["format"] = ext
        result = cloudinary.uploader.upload(str(path), **options)
        return CloudAsset(
            url=result["secure_url"],
            public_id=result["public_id"],
        )
