"""Embedding providers — business logic never calls a vendor SDK directly."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class EmbeddingProvider(Protocol):
    model: str
    dimension: int
    version: str

    def embed(self, text: str) -> list[float]: ...


_TOKEN = re.compile(r"[a-z0-9']+")


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(item * item for item in values)) or 1.0
    return [item / norm for item in values]


class HashingEmbeddingProvider:
    """Deterministic sparse hashing embedder. No network. Dimension is configurable."""

    def __init__(
        self,
        *,
        dimension: int = 384,
        model: str = "hashing-v1",
        version: str = "v1",
    ) -> None:
        if dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be > 0")
        self.dimension = int(dimension)
        self.model = model
        self.version = version

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = _TOKEN.findall((text or "").lower())
        if not tokens:
            vector[0] = 1.0
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, 32, 4):
                idx = int.from_bytes(digest[offset : offset + 4], "big") % self.dimension
                sign = 1.0 if digest[offset] % 2 == 0 else -1.0
                vector[idx] += sign
        return _l2_normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def build_embedding_provider(
    *,
    provider: str = "hashing",
    model: str = "hashing-v1",
    dimension: int = 384,
    version: str = "v1",
) -> EmbeddingProvider:
    name = (provider or "hashing").strip().lower()
    if name in {"hashing", "hash", "local"}:
        return HashingEmbeddingProvider(
            dimension=dimension, model=model or "hashing-v1", version=version or "v1"
        )
    return HashingEmbeddingProvider(
        dimension=dimension, model=model or "hashing-v1", version=version or "v1"
    )
