"""MongoDB — owned collections + indexes. Never touches `users`."""

from __future__ import annotations

from pymongo import MongoClient, ReturnDocument, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import (
    BulkWriteError,
    CollectionInvalid,
    DuplicateKeyError,
    OperationFailure,
)

from core import call_log
from core.db import schema as S


def _oid_or_str(value: str) -> dict | str:
    """Match a hex id stored as either ObjectId or string."""
    from bson import ObjectId

    text = str(value or "").strip()
    if ObjectId.is_valid(text):
        return {"$in": [text, ObjectId(text)]}
    return text


def _without_none(doc: dict) -> dict:
    return {key: value for key, value in doc.items() if value is not None}


def atlas_profile_fields(fields: dict, conversation_id: str) -> dict:
    """Atlas user_profile_memory requires root key + value (collMod is skipped)."""
    payload = _without_none(dict(fields))
    cid = str(conversation_id or "").strip()
    payload["key"] = "profile"
    payload["value"] = {
        "profile": payload.get("profile") or {},
        "facts": payload.get("facts") or [],
    }
    if cid:
        payload["sourceConversationId"] = cid
    return payload


def atlas_learning_fields(fields: dict, conversation_id: str) -> dict:
    """Atlas learning_memory requires root type + string value (collMod is skipped)."""
    payload = _without_none(dict(fields))
    cid = str(conversation_id or "").strip()
    payload["type"] = "learning"
    assessment = payload.get("overallAssessment") or {}
    level = ""
    if isinstance(assessment, dict):
        level = str(assessment.get("level") or "").strip()
    payload["value"] = level or "aggregated"
    if cid:
        payload["sourceConversationId"] = cid
    updated = payload.get("updatedAt")
    if updated is not None:
        payload["lastObservedAt"] = updated
    return payload


class MongoStore:
    def __init__(
        self,
        uri: str,
        db_name: str = "ai_talk",
        *,
        users_db: str = "",
    ) -> None:
        self._client = MongoClient(
            uri,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
            socketTimeoutMS=15000,
        )
        self._db: Database = self._client[db_name]
        users_name = (users_db or "").strip() or db_name
        self.users: Collection = self._client[users_name][S.USERS]
        self.topics: Collection = self._db[S.TOPICS]
        self.topic_progress: Collection = self._db[S.TOPIC_PROGRESS]
        self.conversation_sessions: Collection = self._db[S.CONVERSATION_SESSIONS]
        self.messages: Collection = self._db[S.MESSAGES]
        self.conversation_summaries: Collection = self._db[S.CONVERSATION_SUMMARIES]
        self.user_profile_memory: Collection = self._db[S.USER_PROFILE_MEMORY]
        self.learning_memory: Collection = self._db[S.LEARNING_MEMORY]
        self.memory_metadata: Collection = self._db[S.MEMORY_METADATA]

    def ping(self) -> None:
        self._client.admin.command("ping")

    def ensure_schema(self) -> None:
        """Create collections (except users) + indexes. Idempotent."""
        existing = set(self._db.list_collection_names())
        if S.USERS in existing:
            call_log.info("MONGO", f"users collection present — left untouched")
        for name in S.OWNED_COLLECTIONS:
            validator = S.COLLECTION_VALIDATORS[name]
            if name not in existing:
                try:
                    self._db.create_collection(name, validator=validator)
                    call_log.info("MONGO", f"created collection={name}")
                except CollectionInvalid:
                    pass
            else:
                try:
                    self._db.command(
                        {
                            "collMod": name,
                            "validator": validator,
                            "validationLevel": "moderate",
                        }
                    )
                except OperationFailure as exc:
                    call_log.warn("MONGO", f"validator skip {name}: {exc}")
        self._drop_legacy_profile_key_index()
        self._drop_legacy_learning_indexes()
        for coll_name, keys, unique in S.INDEXES:
            try:
                self._db[coll_name].create_index(keys, unique=unique)
            except Exception as exc:  # noqa: BLE001
                call_log.warn("MONGO", f"index skip {coll_name} {keys}: {exc}")
        print(
            "[api] mongo schema ready: "
            + ", ".join(S.OWNED_COLLECTIONS)
            + " (users ignored)"
        )

    def seed_global_topics(self) -> dict[str, int]:
        """Upsert 25 global topics by slug. Never writes userId. Idempotent."""
        from core.db.topics_seed import TOPICS, utc_now

        slugs = [t["slug"] for t in TOPICS]
        self.topics.delete_many(
            {"$or": [{"slug": {"$exists": False}}, {"slug": {"$nin": slugs}}]}
        )
        now = utc_now()
        inserted = 0
        updated = 0
        for topic in TOPICS:
            payload = {k: v for k, v in topic.items()}
            result = self.topics.update_one(
                {"slug": topic["slug"]},
                {
                    "$set": {**payload, "updatedAt": now},
                    "$setOnInsert": {"createdAt": now},
                    "$unset": {"userId": ""},
                },
                upsert=True,
            )
            if result.upserted_id is not None:
                inserted += 1
            elif result.modified_count:
                updated += 1
        return {
            "inserted": inserted,
            "updated": updated,
            "unchanged": len(TOPICS) - inserted - updated,
            "total": self.topics.count_documents({"slug": {"$in": slugs}}),
        }

    def _drop_legacy_profile_key_index(self) -> None:
        """One profile doc per user — drop the old unique (userId, key) index."""
        try:
            self.user_profile_memory.drop_index("userId_1_key_1")
            call_log.info("MONGO", "dropped legacy user_profile_memory userId_1_key_1")
        except Exception:
            pass

    def _drop_legacy_learning_indexes(self) -> None:
        """One learning-memory doc per user — drop per-type indexes."""
        for name in ("userId_1_type_1", "userId_1_topicId_1", "lastObservedAt_-1"):
            try:
                self.learning_memory.drop_index(name)
                call_log.info("MONGO", f"dropped legacy learning_memory {name}")
            except Exception:
                pass

    def find_user(self, user_id: str) -> dict | None:
        """Read-only lookup. Never creates or writes `users`."""
        from bson import ObjectId

        uid = (user_id or "").strip()
        if not uid:
            return None
        queries: list[dict] = [{"_id": uid}, {"userId": uid}]
        if ObjectId.is_valid(uid):
            queries.insert(0, {"_id": ObjectId(uid)})
        for query in queries:
            doc = self.users.find_one(query)
            if doc:
                return doc
        return None

    def find_in_progress(self, user_id: str) -> dict | None:
        return self.topic_progress.find_one(
            {"userId": user_id, "status": "IN_PROGRESS"}
        )

    def list_progress(self, user_id: str) -> list[dict]:
        return list(self.topic_progress.find({"userId": user_id}))

    def list_active_topics(self, level: str) -> list[dict]:
        return list(
            self.topics.find(
                {"level": level, "isActive": True},
            ).sort("order", 1)
        )

    def list_curriculum_topics(self, level: str | None = None) -> list[dict]:
        query: dict = {"isActive": {"$ne": False}}
        if level:
            query["level"] = str(level).strip().upper()
        return list(self.topics.find(query).sort([("level", 1), ("order", 1)]))

    def find_topic(self, topic_id) -> dict | None:
        from bson import ObjectId

        if topic_id is None:
            return None
        doc = self.topics.find_one({"_id": topic_id})
        if doc:
            return doc
        text = str(topic_id)
        if ObjectId.is_valid(text):
            return self.topics.find_one({"_id": ObjectId(text)})
        return self.topics.find_one({"slug": text})

    def upsert_progress(self, docs: list[dict]) -> None:
        if not docs:
            return
        ops = []
        for doc in docs:
            payload = {key: value for key, value in doc.items() if value is not None}
            ops.append(
                UpdateOne(
                    {"userId": payload["userId"], "topicId": payload["topicId"]},
                    {"$setOnInsert": payload},
                    upsert=True,
                )
            )
        try:
            self.topic_progress.bulk_write(ops, ordered=False)
        except BulkWriteError as exc:
            call_log.error("MONGO", f"upsert_progress failed: {exc.details}")
            raise

    def mark_in_progress(self, user_id: str, topic_id) -> dict | None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        return self.topic_progress.find_one_and_update(
            {"userId": user_id, "topicId": topic_id, "status": "NOT_STARTED"},
            {"$set": {"status": "IN_PROGRESS", "startedAt": now, "updatedAt": now}},
            return_document=ReturnDocument.AFTER,
        )

    def reopen_for_revisit(self, user_id: str, topic_id) -> dict | None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        uid = str(user_id or "").strip()
        if not uid or topic_id is None:
            return None
        return self.topic_progress.find_one_and_update(
            {
                "userId": uid,
                "topicId": topic_id,
                "status": "COMPLETED",
                "needsRevisit": True,
            },
            {
                "$set": {"status": "IN_PROGRESS", "updatedAt": now},
                "$inc": {"attemptCount": 1},
            },
            return_document=ReturnDocument.AFTER,
        )

    def update_progress_fields(self, user_id: str, topic_id, fields: dict) -> dict | None:
        from bson import ObjectId
        from datetime import datetime, timezone

        uid = str(user_id or "").strip()
        if not uid or topic_id is None or not fields:
            return None
        payload = dict(fields)
        payload["updatedAt"] = payload.get("updatedAt") or datetime.now(timezone.utc)
        topic_ids = [topic_id]
        text = str(topic_id)
        if ObjectId.is_valid(text):
            oid = ObjectId(text)
            if oid not in topic_ids:
                topic_ids.append(oid)
        return self.topic_progress.find_one_and_update(
            {"userId": uid, "topicId": {"$in": topic_ids}},
            {"$set": payload},
            return_document=ReturnDocument.AFTER,
        )

    def find_progress(self, user_id: str, topic_id) -> dict | None:
        from bson import ObjectId

        uid = str(user_id or "").strip()
        if not uid or topic_id is None:
            return None
        doc = self.topic_progress.find_one({"userId": uid, "topicId": topic_id})
        if doc:
            return doc
        text = str(topic_id)
        if ObjectId.is_valid(text):
            return self.topic_progress.find_one(
                {"userId": uid, "topicId": ObjectId(text)}
            )
        return None

    def apply_progress_from_conversation(
        self,
        user_id: str,
        topic_id,
        conversation_id: str,
        fields: dict,
    ) -> dict | None:
        from bson import ObjectId

        uid = str(user_id or "").strip()
        cid = str(conversation_id or "").strip()
        if not uid or topic_id is None or not cid:
            return None
        topic_ids: list = [topic_id]
        conv_ids: list = [cid]
        text = str(topic_id)
        if ObjectId.is_valid(text):
            oid = ObjectId(text)
            if oid not in topic_ids:
                topic_ids.append(oid)
        if ObjectId.is_valid(cid):
            conv_ids.append(ObjectId(cid))
        filt = {
            "userId": uid,
            "topicId": {"$in": topic_ids},
            "processedConversationIds": {"$nin": conv_ids},
        }
        payload = dict(fields)
        return self.topic_progress.find_one_and_update(
            filt,
            {
                "$set": payload,
                "$inc": {"attemptCount": 1},
                "$addToSet": {"processedConversationIds": cid},
            },
            return_document=ReturnDocument.AFTER,
        )

    def insert_conversation_session(self, doc: dict) -> dict:
        payload = dict(doc)
        result = self.conversation_sessions.insert_one(payload)
        payload["_id"] = result.inserted_id
        return payload

    def find_conversation_session(self, conversation_id: str) -> dict | None:
        from bson import ObjectId

        cid = str(conversation_id or "").strip()
        if not cid:
            return None
        if ObjectId.is_valid(cid):
            doc = self.conversation_sessions.find_one({"_id": ObjectId(cid)})
            if doc:
                return doc
        return self.conversation_sessions.find_one({"_id": cid})

    def close_conversation_session(
        self,
        conversation_id: str,
        *,
        status: str,
        ended_at,
        duration_seconds: int,
        end_reason: str | None = None,
    ) -> dict | None:
        from datetime import datetime, timezone

        from bson import ObjectId

        cid = str(conversation_id or "").strip()
        if not cid:
            return None
        filt: dict = {"status": "ACTIVE"}
        if ObjectId.is_valid(cid):
            filt["_id"] = ObjectId(cid)
        else:
            filt["_id"] = cid
        now = datetime.now(timezone.utc)
        fields: dict = {
            "status": status,
            "endedAt": ended_at,
            "durationSeconds": duration_seconds,
            "duration": duration_seconds,
            "updatedAt": now,
        }
        if end_reason:
            fields["endReason"] = end_reason
        return self.conversation_sessions.find_one_and_update(
            filt,
            {"$set": fields},
            return_document=ReturnDocument.AFTER,
        )

    def claim_next_message_sequence(
        self, conversation_id: str, *, user_id: str | None = None
    ) -> dict | None:
        from datetime import datetime, timezone

        from bson import ObjectId

        cid = str(conversation_id or "").strip()
        if not cid:
            return None
        filt: dict = {"status": "ACTIVE"}
        if ObjectId.is_valid(cid):
            filt["_id"] = ObjectId(cid)
        else:
            filt["_id"] = cid
        uid = str(user_id or "").strip()
        if uid:
            filt["userId"] = _oid_or_str(uid)
        now = datetime.now(timezone.utc)
        return self.conversation_sessions.find_one_and_update(
            filt,
            {
                "$inc": {"messageCount": 1},
                "$set": {"lastMessageAt": now, "updatedAt": now},
            },
            return_document=ReturnDocument.AFTER,
        )

    def insert_message(self, doc: dict) -> dict:
        payload = dict(doc)
        result = self.messages.insert_one(payload)
        payload["_id"] = result.inserted_id
        return payload

    def list_messages(self, conversation_id: str) -> list[dict]:
        from bson import ObjectId

        cid = str(conversation_id or "").strip()
        if not cid:
            return []
        ids: list = [cid]
        if ObjectId.is_valid(cid):
            ids.append(ObjectId(cid))
        return list(
            self.messages.find({"conversationId": {"$in": ids}}).sort("sequence", 1)
        )

    def find_conversation_summary(self, conversation_id: str) -> dict | None:
        from bson import ObjectId

        cid = str(conversation_id or "").strip()
        if not cid:
            return None
        ids: list = [cid]
        if ObjectId.is_valid(cid):
            ids.append(ObjectId(cid))
        return self.conversation_summaries.find_one({"conversationId": {"$in": ids}})

    def upsert_conversation_summary(self, conversation_id: str, doc: dict) -> dict:
        from datetime import datetime, timezone

        from bson import ObjectId

        cid = str(conversation_id or "").strip()
        payload = dict(doc)
        now = datetime.now(timezone.utc)
        payload["updatedAt"] = payload.get("updatedAt") or now
        filt: dict
        if ObjectId.is_valid(cid):
            filt = {"conversationId": {"$in": [cid, ObjectId(cid)]}}
        else:
            filt = {"conversationId": cid}
        self.conversation_summaries.update_one(
            filt,
            {"$set": payload, "$setOnInsert": {"createdAt": payload.get("createdAt") or now}},
            upsert=True,
        )
        saved = self.find_conversation_summary(cid)
        if saved is None:
            payload.setdefault("createdAt", now)
            return payload
        return saved

    def find_user_profile(self, user_id: str) -> dict | None:
        uid = str(user_id or "").strip()
        if not uid:
            return None
        return self.user_profile_memory.find_one({"userId": uid})

    def apply_profile_from_conversation(
        self,
        user_id: str,
        conversation_id: str,
        fields: dict,
    ) -> dict | None:
        """Atomic upsert of one profile doc per user. No-op if conversation already processed."""
        from datetime import datetime, timezone

        from bson import ObjectId

        uid = str(user_id or "").strip()
        cid = str(conversation_id or "").strip()
        if not uid or not cid:
            return None
        conv_ids: list = [cid]
        if ObjectId.is_valid(cid):
            conv_ids.append(ObjectId(cid))
        now = datetime.now(timezone.utc)
        payload = atlas_profile_fields(fields, cid)
        payload.pop("createdAt", None)
        payload.pop("version", None)
        payload.pop("userId", None)
        payload.pop("processedConversationIds", None)
        filt = {
            "userId": uid,
            "processedConversationIds": {"$nin": conv_ids},
        }
        update = {
            "$set": payload,
            "$inc": {"version": 1},
            "$addToSet": {"processedConversationIds": cid},
            "$setOnInsert": {"userId": uid, "createdAt": now},
        }

        def _apply(*, upsert: bool) -> dict | None:
            return self.user_profile_memory.find_one_and_update(
                filt,
                update if upsert else {
                    "$set": payload,
                    "$inc": {"version": 1},
                    "$addToSet": {"processedConversationIds": cid},
                },
                upsert=upsert,
                return_document=ReturnDocument.AFTER,
            )

        try:
            return _apply(upsert=True)
        except DuplicateKeyError:
            return _apply(upsert=False)
        except OperationFailure as exc:
            call_log.error("MONGO", f"profile upsert failed: {exc.details or exc}")
            raise

    def find_learning_memory(self, user_id: str) -> dict | None:
        uid = str(user_id or "").strip()
        if not uid:
            return None
        return self.learning_memory.find_one({"userId": uid})

    def apply_learning_memory_from_conversation(
        self,
        user_id: str,
        conversation_id: str,
        fields: dict,
    ) -> dict | None:
        """Atomic upsert of one learning-memory doc per user. No-op if already processed."""
        from datetime import datetime, timezone

        from bson import ObjectId

        uid = str(user_id or "").strip()
        cid = str(conversation_id or "").strip()
        if not uid or not cid:
            return None
        conv_ids: list = [cid]
        if ObjectId.is_valid(cid):
            conv_ids.append(ObjectId(cid))
        now = datetime.now(timezone.utc)
        payload = atlas_learning_fields(fields, cid)
        payload.pop("createdAt", None)
        payload.pop("version", None)
        payload.pop("userId", None)
        payload.pop("processedConversationIds", None)
        filt = {
            "userId": uid,
            "processedConversationIds": {"$nin": conv_ids},
        }
        update = {
            "$set": payload,
            "$inc": {"version": 1},
            "$addToSet": {"processedConversationIds": cid},
            "$setOnInsert": {"userId": uid, "createdAt": now},
        }

        def _apply(*, upsert: bool) -> dict | None:
            return self.learning_memory.find_one_and_update(
                filt,
                update if upsert else {
                    "$set": payload,
                    "$inc": {"version": 1},
                    "$addToSet": {"processedConversationIds": cid},
                },
                upsert=upsert,
                return_document=ReturnDocument.AFTER,
            )

        try:
            return _apply(upsert=True)
        except DuplicateKeyError:
            return _apply(upsert=False)
        except OperationFailure as exc:
            call_log.error("MONGO", f"learning upsert failed: {exc.details or exc}")
            raise

    def find_memory_metadata(self, memory_id: str) -> dict | None:
        mid = str(memory_id or "").strip()
        if not mid:
            return None
        return self.memory_metadata.find_one({"memoryId": mid})

    def find_memory_by_identity(
        self, *, tenant_id: str, user_id: str, identity_key: str
    ) -> dict | None:
        return self.memory_metadata.find_one(
            {
                "tenantId": str(tenant_id or "").strip(),
                "userId": str(user_id or "").strip(),
                "identityKey": str(identity_key or "").strip(),
            }
        )

    def list_memory_metadata(
        self, *, tenant_id: str, user_id: str, indexed_only: bool | None = None
    ) -> list[dict]:
        filt: dict = {
            "tenantId": str(tenant_id or "").strip(),
            "userId": str(user_id or "").strip(),
        }
        if indexed_only is True:
            filt["indexed"] = True
        elif indexed_only is False:
            filt["indexed"] = {"$ne": True}
        return list(self.memory_metadata.find(filt))

    def list_memory_metadata_for_tenant(self, tenant_id: str) -> list[dict]:
        return list(
            self.memory_metadata.find({"tenantId": str(tenant_id or "").strip()})
        )

    def upsert_memory_metadata(self, doc: dict) -> dict:
        from datetime import datetime, timezone

        payload = dict(doc)
        now = datetime.now(timezone.utc)
        payload["updatedAt"] = payload.get("updatedAt") or now
        created_at = payload.pop("createdAt", None) or now
        memory_id = str(payload.get("memoryId") or "").strip()
        identity = str(payload.get("identityKey") or "").strip()
        if memory_id:
            filt = {"memoryId": memory_id}
        else:
            filt = {
                "tenantId": payload.get("tenantId"),
                "userId": payload.get("userId"),
                "identityKey": identity,
            }
        self.memory_metadata.update_one(
            filt,
            {
                "$set": payload,
                "$setOnInsert": {"createdAt": created_at},
            },
            upsert=True,
        )
        return self.memory_metadata.find_one(filt) or payload

    def delete_memory_metadata(self, memory_id: str) -> None:
        mid = str(memory_id or "").strip()
        if not mid:
            return
        self.memory_metadata.delete_one({"memoryId": mid})

    def delete_user_memory_metadata(self, *, tenant_id: str, user_id: str) -> None:
        self.memory_metadata.delete_many(
            {
                "tenantId": str(tenant_id or "").strip(),
                "userId": str(user_id or "").strip(),
            }
        )
