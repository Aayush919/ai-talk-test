"""In-memory Qdrant stand-in for tests. Production uses wrappers.qdrant_store.QdrantStore."""

from __future__ import annotations

from typing import Any, Protocol

from core.semantic.embeddings import cosine_similarity


class SemanticMemoryRepository(Protocol):
    def available(self) -> bool: ...
    def healthCheck(self) -> str: ...
    def createCollection(self) -> None: ...
    def upsertMemory(self, *, point_id: str, vector: list[float], payload: dict[str, Any]) -> None: ...
    def updateMemory(self, *, point_id: str, vector: list[float], payload: dict[str, Any]) -> None: ...
    def searchMemories(
        self,
        *,
        vector: list[float],
        tenant_id: str,
        user_id: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
    def getMemoryById(self, point_id: str) -> dict[str, Any] | None: ...
    def deleteMemory(self, point_id: str) -> None: ...
    def deleteUserMemories(self, *, tenant_id: str, user_id: str) -> None: ...
    def deleteTenantMemories(self, *, tenant_id: str) -> None: ...


class InMemorySemanticRepository:
    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.online = True

    def available(self) -> bool:
        return self.online

    def healthCheck(self) -> str:
        return "QDRANT_AVAILABLE" if self.online else "QDRANT_UNAVAILABLE"

    def createCollection(self) -> None:
        return None

    def updateMemory(self, *, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.upsertMemory(point_id=point_id, vector=vector, payload=payload)

    def upsertMemory(self, *, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        if not self.online:
            raise RuntimeError("QDRANT_UNAVAILABLE")
        self.points[str(point_id)] = {
            "id": str(point_id),
            "vector": list(vector),
            "payload": dict(payload),
        }

    def searchMemories(
        self,
        *,
        vector: list[float],
        tenant_id: str,
        user_id: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.online:
            raise RuntimeError("QDRANT_UNAVAILABLE")
        extra = filters or {}
        scored: list[tuple[float, dict[str, Any]]] = []
        for point in self.points.values():
            payload = point.get("payload") or {}
            if str(payload.get("tenantId") or "") != str(tenant_id):
                continue
            if str(payload.get("userId") or "") != str(user_id):
                continue
            skip = False
            for key in ("memoryType", "category", "skill", "topicId", "topicLevel", "status", "scope"):
                wanted = extra.get(key)
                if wanted is None or wanted == "":
                    continue
                if str(payload.get(key) or "") != str(wanted):
                    skip = True
                    break
            if skip:
                continue
            score = cosine_similarity(vector, point.get("vector") or [])
            scored.append((score, {"id": point["id"], "score": score, "payload": dict(payload)}))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[: max(1, int(limit))]]

    def getMemoryById(self, point_id: str) -> dict[str, Any] | None:
        point = self.points.get(str(point_id))
        if not point:
            return None
        return {"id": point["id"], "payload": dict(point["payload"])}

    def deleteMemory(self, point_id: str) -> None:
        self.points.pop(str(point_id), None)

    def deleteUserMemories(self, *, tenant_id: str, user_id: str) -> None:
        drop = [
            key
            for key, point in self.points.items()
            if str((point.get("payload") or {}).get("tenantId")) == str(tenant_id)
            and str((point.get("payload") or {}).get("userId")) == str(user_id)
        ]
        for key in drop:
            self.points.pop(key, None)

    def deleteTenantMemories(self, *, tenant_id: str) -> None:
        drop = [
            key
            for key, point in self.points.items()
            if str((point.get("payload") or {}).get("tenantId")) == str(tenant_id)
        ]
        for key in drop:
            self.points.pop(key, None)
