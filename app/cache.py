"""Two-layer cache: in-process LRU (fast) in front of Postgres/Supabase (persistent).

Postgres is best-effort: any DB failure degrades to memory-only, never breaks a request.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from typing import Any

log = logging.getLogger("healthai.cache")


class LRUCache:
    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"size": len(self._data), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}


class PostgresStore:
    """Tiny key/value table in Supabase Postgres. Disabled when no DATABASE_URL."""

    DDL = """CREATE TABLE IF NOT EXISTS query_cache (
        key TEXT PRIMARY KEY, value JSONB NOT NULL,
        hit_count INTEGER NOT NULL DEFAULT 0, created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""

    def __init__(self, database_url: str):
        self.enabled = False
        self._url = database_url
        if not database_url:
            return
        try:
            import psycopg  # imported lazily so tests/dev work without a DB driver
            self._psycopg = psycopg
            with psycopg.connect(database_url, connect_timeout=5) as conn:
                conn.execute(self.DDL)
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres cache disabled: %s", type(exc).__name__)

    def get(self, key: str):
        if not self.enabled:
            return None
        try:
            with self._psycopg.connect(self._url, connect_timeout=3) as conn:
                row = conn.execute(
                    "UPDATE query_cache SET hit_count = hit_count + 1 WHERE key=%s RETURNING value", (key,)
                ).fetchone()
            return row[0] if row else None
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres get failed: %s", type(exc).__name__)
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            with self._psycopg.connect(self._url, connect_timeout=3) as conn:
                conn.execute(
                    "INSERT INTO query_cache(key, value) VALUES (%s, %s::jsonb) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (key, json.dumps(value)),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Postgres set failed: %s", type(exc).__name__)


class TieredCache:
    def __init__(self, memory: LRUCache, store: PostgresStore | None = None):
        self.memory = memory
        self.store = store

    def get(self, key: str):
        val = self.memory.get(key)
        if val is not None:
            return val
        if self.store:
            val = self.store.get(key)
            if val is not None:
                self.memory.set(key, val)  # promote to L1
                return val
        return None

    def set(self, key: str, value: Any) -> None:
        self.memory.set(key, value)
        if self.store:
            self.store.set(key, value)
