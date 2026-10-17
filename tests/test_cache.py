from app.cache import LRUCache, PostgresStore, TieredCache


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


class FakeStore:
    enabled = True
    def __init__(self): self.data = {}
    def get(self, k): return self.data.get(k)
    def set(self, k, v): self.data[k] = v


def test_tiered_cache_promotes_from_store_to_memory():
    store = FakeStore(); store.data["k"] = {"x": 1}
    t = TieredCache(LRUCache(), store)
    assert t.get("k") == {"x": 1}
    store.data.clear()               # now only L1 has it
    assert t.get("k") == {"x": 1}


def test_tiered_cache_writes_through():
    store = FakeStore()
    TieredCache(LRUCache(), store).set("k", [1, 2])
    assert store.data["k"] == [1, 2]


def test_postgres_store_disabled_without_url_is_noop():
    s = PostgresStore("")
    assert s.enabled is False
    assert s.get("k") is None
    s.set("k", {"a": 1})   # must not raise
