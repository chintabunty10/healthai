"""Two-layer cache built to survive load and a flaky database.

L1  in-process LRU            microseconds, thread-safe
L2  Supabase/Postgres         persistent, shared across workers/restarts

How L2 stays off the critical path / cannot take the app down:
  * connection POOL (psycopg_pool) - no connect-per-call, bounded connections to Supabase
  * writes (and hit counters) go through a bounded queue to ONE background thread (write-behind);
    the request never waits for them, and when the queue is full we drop the write (it's a cache)
  * reads are bounded: pool wait timeout + server statement_timeout
  * a CIRCUIT BREAKER stops touching the DB after repeated failures, then probes it again later
  * every failure degrades to "L1 only"; nothing in here raises into a request
"""
from __future__ import annotations

import json
import logging
import queue
import threading
from collections import OrderedDict
from typing import Any

from .metrics import CACHE_BREAKER_STATE, CACHE_DB_EVENTS, CACHE_LOOKUPS, CACHE_WRITE_QUEUE
from .resilience import CircuitBreaker

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


class PostgresBackend:
    """Thin psycopg pool wrapper. Raises on failure; ResilientStore decides what that means."""

    DDL = """CREATE TABLE IF NOT EXISTS query_cache (
        key TEXT PRIMARY KEY, value JSONB NOT NULL,
        hit_count INTEGER NOT NULL DEFAULT 0, created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""

    def __init__(self, database_url: str, *, min_size=1, max_size=4, pool_timeout=1.0, statement_timeout_ms=1500):
        from psycopg_pool import ConnectionPool  # lazy: app and tests run without a DB driver

        self.statement_timeout_ms = int(statement_timeout_ms)
        self.pool = ConnectionPool(
            database_url, min_size=min_size, max_size=max_size, timeout=pool_timeout,
            open=False, name="healthai-cache",
            # prepare_threshold=None: safe behind Supabase's pgbouncer-style pooler (no prepared statements)
            kwargs={"connect_timeout": 3, "prepare_threshold": None},
        )
        self.pool.open(wait=False)  # connect in the background; never block startup

    def _run(self, sql: str, params=(), fetch: bool = False):
        with self.pool.connection() as conn:  # waits at most pool_timeout, else PoolTimeout
            conn.execute(f"SET LOCAL statement_timeout = {self.statement_timeout_ms}")
            cur = conn.execute(sql, params)
            return cur.fetchone() if fetch else None

    def ensure_schema(self) -> None:
        self._run(self.DDL)

    def read(self, key: str):
        row = self._run("SELECT value FROM query_cache WHERE key=%s", (key,), fetch=True)
        return row[0] if row else None

    def write(self, key: str, value: Any) -> None:
        self._run("INSERT INTO query_cache(key, value) VALUES (%s, %s::jsonb) "
                  "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (key, json.dumps(value)))

    def touch(self, key: str) -> None:
        self._run("UPDATE query_cache SET hit_count = hit_count + 1 WHERE key=%s", (key,))

    def close(self) -> None:
        self.pool.close()


class ResilientStore:
    """L2 wrapper: breaker + write-behind queue around any backend with
    ensure_schema/read/write/touch/close (PostgresBackend in production, a fake in tests)."""

    def __init__(self, backend, breaker: CircuitBreaker | None = None, queue_size: int = 1000,
                 schema_retry_seconds: float = 5.0):
        self.backend = backend
        self.breaker = breaker or CircuitBreaker()
        self.dropped_writes = 0
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._ready, self._stop = threading.Event(), threading.Event()
        self._retry = schema_retry_seconds
        self._worker = threading.Thread(target=self._run, name="cache-writer", daemon=True)
        self._worker.start()

    # ---- state -------------------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    def status(self) -> dict:
        return {"ready": self.ready, "breaker": self.breaker.state,
                "queue_depth": self._queue.qsize(), "dropped_writes": self.dropped_writes}

    def _publish(self) -> None:
        CACHE_BREAKER_STATE.set({"closed": 0, "half_open": 1, "open": 2}[self.breaker.state])
        CACHE_WRITE_QUEUE.set(self._queue.qsize())

    def _failed(self, op: str, exc: Exception) -> None:
        self.breaker.record_failure()
        CACHE_DB_EVENTS.inc(event="error")
        log.warning("cache_db_error", extra={"op": op, "error": type(exc).__name__, "breaker": self.breaker.state})

    # ---- request-path API (never raises, never blocks on writes) ------------------------------
    def get(self, key: str):
        if not self.ready:
            return None
        if not self.breaker.allow():
            CACHE_DB_EVENTS.inc(event="skipped_open_circuit")
            return None
        try:
            value = self.backend.read(key)
        except Exception as exc:  # noqa: BLE001
            self._failed("get", exc)
            self._publish()
            return None
        self.breaker.record_success()
        self._publish()
        if value is not None:
            self._enqueue(("touch", key))
        return value

    def set(self, key: str, value: Any) -> None:
        self._enqueue(("set", key, value))

    def _enqueue(self, item: tuple) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped_writes += 1
            CACHE_DB_EVENTS.inc(event="dropped_write")
        CACHE_WRITE_QUEUE.set(self._queue.qsize())

    # ---- background writer ---------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.backend.ensure_schema()
                self._ready.set()
                log.info("cache_db_ready")
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("cache_db_schema_failed", extra={"error": type(exc).__name__})
                self._stop.wait(self._retry)
        while not self._stop.is_set() or self._queue.unfinished_tasks:
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._handle(item)
            finally:
                self._queue.task_done()

    def _handle(self, item: tuple) -> None:
        if not self.ready or not self.breaker.allow():
            CACHE_DB_EVENTS.inc(event="skipped_open_circuit")
            return
        try:
            if item[0] == "set":
                self.backend.write(item[1], item[2])
                CACHE_DB_EVENTS.inc(event="write_ok")
            else:
                self.backend.touch(item[1])
            self.breaker.record_success()
        except Exception as exc:  # noqa: BLE001
            self._failed(item[0], exc)
        self._publish()

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until queued writes are processed (used by tests and shutdown)."""
        import time
        end = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.01)
        return not self._queue.unfinished_tasks

    def close(self) -> None:
        self.flush(2.0)
        self._stop.set()
        self._worker.join(timeout=3.0)
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001
            pass


def make_store(database_url: str, *, min_size=1, max_size=4, pool_timeout=1.0, statement_timeout_ms=1500,
               queue_size=1000, breaker_failures=3, breaker_cooldown=30.0) -> ResilientStore | None:
    """Returns None when no DATABASE_URL is configured or the driver is unavailable (memory-only mode)."""
    if not database_url:
        return None
    try:
        backend = PostgresBackend(database_url, min_size=min_size, max_size=max_size,
                                  pool_timeout=pool_timeout, statement_timeout_ms=statement_timeout_ms)
    except Exception as exc:  # noqa: BLE001
        log.warning("cache_db_disabled", extra={"error": type(exc).__name__})
        return None
    return ResilientStore(backend, CircuitBreaker(breaker_failures, breaker_cooldown), queue_size)


class TieredCache:
    def __init__(self, memory: LRUCache, store: ResilientStore | None = None):
        self.memory = memory
        self.store = store

    def get(self, key: str, l2: bool = True):
        """l2=False for cheap-to-recompute values: a network round trip to a remote DB can cost
        more than recomputing, so those live in memory only."""
        kind = key.split(":", 1)[0]
        value = self.memory.get(key)
        if value is not None:
            CACHE_LOOKUPS.inc(kind=kind, result="l1")
            return value
        if self.store and l2:
            value = self.store.get(key)
            if value is not None:
                self.memory.set(key, value)  # promote to L1
                CACHE_LOOKUPS.inc(kind=kind, result="l2")
                return value
        CACHE_LOOKUPS.inc(kind=kind, result="miss")
        return None

    def set(self, key: str, value: Any, l2: bool = True) -> None:
        self.memory.set(key, value)  # immediate
        if self.store and l2:
            self.store.set(key, value)  # asynchronous write-behind

    def status(self) -> dict:
        out = {"l1": self.memory.stats()}
        if self.store:
            out["l2"] = self.store.status()
        return out
