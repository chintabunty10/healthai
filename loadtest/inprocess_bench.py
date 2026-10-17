"""In-process benchmark of the request path (no HTTP): preprocess -> cache -> model -> cache.set.

Purpose: isolate what the cache/DB layer does to latency under 100 concurrent callers, comparing
  * "original": synchronous L2 reads/writes in the request path, every identical miss recomputes
  * "current" : write-behind + circuit breaker + single-flight for identical concurrent misses
across database conditions. This is NOT an HTTP test (use locustfile.py for that) and numbers depend
on the machine; compare scenarios against each other, not against other computers.

Run:  python -m loadtest.inprocess_bench [--users 100 --requests 30]
"""
from __future__ import annotations

import argparse
import random
import statistics
import threading
import time

from app.cache import LRUCache, ResilientStore, TieredCache
from app.model import SymptomModel
from app.resilience import CircuitBreaker, ThreadSingleFlight

COMMON = ["fever, cough", "itching, skin rash", "headache, vomiting", "chest pain, breathlessness"]
POOL = ["fever", "cough", "headache", "fatigue", "vomiting", "chills", "skin rash", "joint pain",
        "abdominal pain", "dizziness", "nausea", "muscle pain", "itching", "sweating"]


class FakeDB:
    """Stands in for Supabase: fixed latency, optional hard failure (fails after `latency`, like a timeout)."""
    def __init__(self, latency=0.03, fail=False):
        self.latency, self.fail, self.calls, self.data = latency, fail, 0, {}
        self._lock = threading.Lock()
    def _io(self):
        with self._lock:
            self.calls += 1
        time.sleep(self.latency)
        if self.fail:
            raise RuntimeError("db down")
    def ensure_schema(self): pass
    def read(self, k): self._io(); return self.data.get(k)
    def write(self, k, v): self._io(); self.data[k] = v
    def touch(self, k): self._io()
    def close(self): pass


class NaiveTiered:
    """Original design: L2 touched synchronously on the request path; errors swallowed but still paid for."""
    def __init__(self, db): self.memory, self.db = LRUCache(1000), db
    def get(self, key):
        v = self.memory.get(key)
        if v is not None: return v
        try: v = self.db.read(key)
        except Exception: return None
        if v is not None: self.memory.set(key, v)
        return v
    def set(self, key, value):
        self.memory.set(key, value)
        try: self.db.write(key, value)
        except Exception: pass


FLIGHT = ThreadSingleFlight()


def handle(cache, model, text, coalesce=False):
    pre = model.preprocessor.process(text)
    key = "predict:" + pre.cache_key
    kw = {"l2": False} if coalesce else {}  # "current" design: predictions skip the DB tier
    if cache.get(key, **kw) is None:
        def compute():
            top = model.predict(pre.vector, 5)
            cache.set(key, [c.name for c in top], **kw)
        FLIGHT.do(key, compute) if coalesce else compute()


def run(label, cache, model, users, per_user, rng_seed=1, coalesce=False):
    lat, errors = [], [0]
    lock = threading.Lock()
    def user(n):
        rng = random.Random(rng_seed + n)
        mine = []
        for _ in range(per_user):
            text = rng.choice(COMMON) if rng.random() < 0.7 else ", ".join(rng.sample(POOL, 3))
            t0 = time.perf_counter()
            try: handle(cache, model, text, coalesce)
            except Exception:
                with lock: errors[0] += 1
            mine.append((time.perf_counter() - t0) * 1000)
        with lock: lat.extend(mine)
    t0 = time.perf_counter()
    ts = [threading.Thread(target=user, args=(i,)) for i in range(users)]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))]
    print(f"{label:<34} rps={len(lat) / wall:7.0f}  p50={q(.5):7.1f}  p95={q(.95):7.1f}  p99={q(.99):7.1f}  max={lat[-1]:7.1f} ms  errors={errors[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--requests", type=int, default=30)
    a = ap.parse_args()
    model = SymptomModel()
    for case, db in [("db healthy (30 ms)", FakeDB(0.03)), ("db slow (500 ms)", FakeDB(0.5)), ("db down (50 ms timeout)", FakeDB(0.05, fail=True))]:
        run(f"original  | {case}", NaiveTiered(db), model, a.users, a.requests)
        db2 = FakeDB(db.latency, db.fail)
        store = ResilientStore(db2, CircuitBreaker(3, 30), queue_size=1000)
        store.wait_ready(2)
        run(f"current   | {case}", TieredCache(LRUCache(1000), store), model, a.users, a.requests, coalesce=True)
        st = store.status(); store.close()
        print(f"{'':<34} -> db calls {db.calls} original vs {db2.calls} current | breaker={st['breaker']} dropped_writes={st['dropped_writes']}")
    run("current   | memory-only (no database)", TieredCache(LRUCache(1000)), model, a.users, a.requests, coalesce=True)


if __name__ == "__main__":
    main()
