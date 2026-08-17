"""Qdrant repository — one shared collection, tenantId+userId on every point."""

from __future__ import annotations

from typing import Any

from core import call_log
from core.db.schema import QDRANT_COLLECTION_DEFAULT, QDRANT_VECTOR_SIZE


def _distance(name: str):
    from qdrant_client.models import Distance

    mapping = {
        "COSINE": Distance.COSINE,
        "EUCLID": Distance.EUCLID,
        "DOT": Distance.DOT,
    }
    return mapping.get((name or "COSINE").upper(), Distance.COSINE)


class QdrantStore:
    def __init__(
        self,
        url: str,
        *,
        api_key: str = "",
        collection: str = QDRANT_COLLECTION_DEFAULT,
        vector_size: int = QDRANT_VECTOR_SIZE,
        distance: str = "COSINE",
    ) -> None:
        self.url = url
        self.collection = collection or QDRANT_COLLECTION_DEFAULT
        self.vector_size = int(vector_size or QDRANT_VECTOR_SIZE)
        self.distance = (distance or "COSINE").upper()
        self._client = None
        if not url:
            call_log.warn("VECTOR", "QDRANT_URL empty — collection not created yet")
            return
        try:
            from qdrant_client import QdrantClient

            kwargs: dict = {"url": url, "timeout": 8}
            if api_key:
                kwargs["api_key"] = api_key
            self._client = QdrantClient(**kwargs)
        except Exception as exc:  # noqa: BLE001
            call_log.warn("VECTOR", f"qdrant client off: {exc}")
            self._client = None

    def available(self) -> bool:
        return self._client is not None

    def healthCheck(self) -> str:
        if self._client is None:
            return "QDRANT_UNAVAILABLE"
        try:
            self._client.get_collections()
            return "QDRANT_AVAILABLE"
        except Exception as exc:  # noqa: BLE001
            call_log.warn("VECTOR", f"qdrant_health_failed: {exc}")
            return "QDRANT_UNAVAILABLE"

    def createCollection(self) -> None:
        self.ensure_collection()

    def ensure_collection(self) -> None:
        if self._client is None:
            return
        from qdrant_client.models import PayloadSchemaType, VectorParams

        names = [c.name for c in self._client.get_collections().collections]
        if self.collection not in names:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=_distance(self.distance),
                ),
            )
            call_log.info(
                "VECTOR",
                f"created collection={self.collection} dim={self.vector_size} {self.distance}",
            )
        for field in (
            "userId",
            "tenantId",
            "memoryType",
            "topicId",
            "category",
            "skill",
            "status",
            "scope",
            "sourceId",
            "memoryId",
        ):
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass
        print(
            f"[api] qdrant ready collection={self.collection} "
            f"dim={self.vector_size} (no Mongo dump, no progress fields)"
        )

    def updateMemory(self, *, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        self.upsertMemory(point_id=point_id, vector=vector, payload=payload)

    def upsertMemory(self, *, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
        if self._client is None:
            raise RuntimeError("QDRANT_UNAVAILABLE")
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=point_id, vector=vector, payload=dict(payload))],
        )

    def searchMemories(
        self,
        *,
        vector: list[float],
        tenant_id: str,
        user_id: str,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._client is None:
            raise RuntimeError("QDRANT_UNAVAILABLE")
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [
            FieldCondition(key="tenantId", match=MatchValue(value=str(tenant_id))),
            FieldCondition(key="userId", match=MatchValue(value=str(user_id))),
        ]
        extra = filters or {}
        for key in ("memoryType", "category", "skill", "topicId", "topicLevel", "status", "scope"):
            value = extra.get(key)
            if value is None or value == "":
                continue
            must.append(FieldCondition(key=key, match=MatchValue(value=str(value))))
        query_filter = Filter(must=must)
        try:
            results = self._client.query_points(
                collection_name=self.collection,
                query=vector,
                query_filter=query_filter,
                limit=max(1, int(limit)),
                with_payload=True,
            )
            points = getattr(results, "points", None) or []
        except Exception:
            points = self._client.search(
                collection_name=self.collection,
                query_vector=vector,
                query_filter=query_filter,
                limit=max(1, int(limit)),
                with_payload=True,
            )
        out: list[dict[str, Any]] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            out.append(
                {
                    "id": str(getattr(point, "id", "")),
                    "score": float(getattr(point, "score", 0) or 0),
                    "payload": payload,
                }
            )
        return out

    def getMemoryById(self, point_id: str) -> dict[str, Any] | None:
        if self._client is None:
            raise RuntimeError("QDRANT_UNAVAILABLE")
        points = self._client.retrieve(
            collection_name=self.collection, ids=[point_id], with_payload=True
        )
        if not points:
            return None
        point = points[0]
        return {
            "id": str(point.id),
            "payload": dict(point.payload or {}),
        }

    def deleteMemory(self, point_id: str) -> None:
        if self._client is None:
            raise RuntimeError("QDRANT_UNAVAILABLE")
        self._client.delete(
            collection_name=self.collection,
            points_selector=[point_id],
        )

    def deleteUserMemories(self, *, tenant_id: str, user_id: str) -> None:
        if self._client is None:
            raise RuntimeError("QDRANT_UNAVAILABLE")
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="tenantId", match=MatchValue(value=str(tenant_id))),
                    FieldCondition(key="userId", match=MatchValue(value=str(user_id))),
                ]
            ),
        )

    def deleteTenantMemories(self, *, tenant_id: str) -> None:
        if self._client is None:
            raise RuntimeError("QDRANT_UNAVAILABLE")
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="tenantId", match=MatchValue(value=str(tenant_id))),
                ]
            ),
        )
