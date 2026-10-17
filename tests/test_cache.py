import threading
import time

from app.cache import LRUCache, ResilientStore, TieredCache, make_store
from app.resilience import CircuitBreaker


def test_lru_evicts_least_recently_used():
    c = LRUCache(max_items=2)
    c.set("a", 1); c.set("b", 2)
    assert c.get("a") == 1          # touch a -> b is now oldest
    c.set("c", 3)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3


def test_lru_stats_track_hits_and_misses():
    c = LRUCache()
    c.set("k", "v")
    c.get("k"); c.get("nope")
    s = c.stats()
    assert (s["hits"], s["misses"], s["hit_rate"]) == (1, 1, 0.5)


class FakeBackend:
    """In-memory stand-in for PostgresBackend with injectable latency / failure."""
    def __init__(self, latency=0.0, fail=False):
        self.data, self.latency, self.fail = {}, latency, fail
        self.reads = self.writes = self.touches = 0
        self.schema_ok = True
    def ensure_schema(self):
        if not self.schema_ok:
            raise RuntimeError("no schema yet")
    def _maybe(self):
        if self.latency: time.sleep(self.latency)
        if self.fail: raise RuntimeError("db down")
    def read(self, k): self.reads += 1; self._maybe(); return self.data.get(k)
    def write(self, k, v): self.writes += 1; self._maybe(); self.data[k] = v
    def touch(self, k): self.touches += 1; self._maybe()
    def close(self): pass


def store_for(backend, **kw):
    s = ResilientStore(backend, schema_retry_seconds=0.05, **kw)
    assert s.wait_ready(2)
    return s


def test_tiered_cache_promotes_from_store_to_memory():
    backend = FakeBackend(); backend.data["k"] = {"x": 1}
    t = TieredCache(LRUCache(), store_for(backend))
    assert t.get("k") == {"x": 1}
    backend.data.clear()             # now only L1 has it
    assert t.get("k") == {"x": 1}


def test_tiered_cache_writes_through_asynchronously():
    backend = FakeBackend()
    store = store_for(backend)
    TieredCache(LRUCache(), store).set("k", [1, 2])
    assert store.flush()
    assert backend.data["k"] == [1, 2]


def test_make_store_is_none_without_database_url():
    assert make_store("") is None


def test_writes_never_block_the_request_even_with_a_slow_database():
    backend = FakeBackend(latency=0.2)            # every DB write takes 200 ms
    store = store_for(backend, queue_size=1000)
    cache = TieredCache(LRUCache(), store)
    start = time.perf_counter()
    for i in range(20):
        cache.set(f"k{i}", i)                      # synchronous version would take ~4 s
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, elapsed
    assert cache.get("k5") == 5                    # served from L1 immediately


def test_full_write_queue_drops_instead_of_blocking_or_raising():
    backend = FakeBackend(latency=0.3)
    store = store_for(backend, queue_size=2)
    for i in range(10):
        store.set(f"k{i}", i)                      # must not raise or block
    assert store.dropped_writes >= 5


def test_breaker_stops_hammering_a_failing_database_then_recovers():
    clock = [0.0]
    backend = FakeBackend(fail=True)
    breaker = CircuitBreaker(failure_threshold=3, cooldown=30, clock=lambda: clock[0])
    store = store_for(backend, breaker=breaker)
    for _ in range(10):
        assert store.get("k") is None              # failures never raise into the caller
    assert backend.reads == 3                      # breaker opened after 3, remaining 7 were skipped
    assert store.status()["breaker"] == "open"
    clock[0] = 31; backend.fail = False; backend.data["k"] = "back"
    assert store.get("k") == "back"                # half-open trial succeeds
    assert store.status()["breaker"] == "closed"


def test_reads_are_skipped_until_schema_is_ready_and_app_still_works():
    backend = FakeBackend(); backend.schema_ok = False
    store = ResilientStore(backend, schema_retry_seconds=0.05)
    cache = TieredCache(LRUCache(), store)
    assert store.ready is False
    assert cache.get("nothing") is None and backend.reads == 0
    cache.set("a", 1); assert cache.get("a") == 1  # L1 keeps working while DB is unavailable
    backend.schema_ok = True
    assert store.wait_ready(2)


def test_hits_are_counted_in_background():
    backend = FakeBackend(); backend.data["k"] = 1
    store = store_for(backend)
    assert store.get("k") == 1 and store.flush()
    assert backend.touches == 1


def test_concurrent_access_is_safe():
    backend = FakeBackend()
    cache = TieredCache(LRUCache(500), store_for(backend))
    def worker(n):
        for i in range(200):
            cache.set(f"{n}:{i}", i); cache.get(f"{n}:{i}")
    ts = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert cache.store.flush(5)
