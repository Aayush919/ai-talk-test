"""Idempotent global topic seed. Does not touch users or topic_progress rows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

from wrappers.mongo_store import MongoStore  # noqa: E402


def main() -> int:
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        print("MONGODB_URI missing")
        return 1
    db = (os.getenv("MONGODB_DB") or "ai_talk").strip() or "ai_talk"
    mongo = MongoStore(uri, db)
    mongo.ping()
    mongo.ensure_schema()
    stats = mongo.seed_global_topics()
    print(
        "topics seed "
        f"inserted={stats['inserted']} updated={stats['updated']} "
        f"unchanged={stats['unchanged']} total={stats['total']}"
    )
    if stats["total"] != 25:
        print(f"expected 25 global topics, got {stats['total']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
