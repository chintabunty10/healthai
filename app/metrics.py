"""Dependency-free, thread-safe metrics with Prometheus text exposition.

Counter / Gauge / Histogram + a Registry. Chosen over `prometheus_client` so tests run anywhere and
the whole mechanism is readable in one file; the exposition format is the same, so Prometheus/Grafana
can scrape /metrics unchanged (swapping in prometheus_client later is a drop-in change).
"""
from __future__ import annotations

import bisect
import threading

DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


def _key(labels: dict) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(pairs) -> str:
    pairs = list(pairs)
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in pairs) + "}" if pairs else ""


class _Metric:
    kind = ""

    def __init__(self, name: str, help_: str):
        self.name, self.help = name, help_
        self._lock = threading.Lock()

    def _header(self) -> list[str]:
        return [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} {self.kind}"]


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name, help_):
        super().__init__(name, help_)
        self._v: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **labels):
        k = _key(labels)
        with self._lock:
            self._v[k] = self._v.get(k, 0.0) + amount

    def value(self, **labels) -> float:
        with self._lock:
            return self._v.get(_key(labels), 0.0)

    def clear(self):
        with self._lock:
            self._v.clear()

    def render(self) -> list[str]:
        with self._lock:
            items = sorted(self._v.items())
        return self._header() + [f"{self.name}{_fmt(k)} {v:g}" for k, v in items]

    def snapshot(self):
        with self._lock:
            return [{"labels": dict(k), "value": v} for k, v in sorted(self._v.items())]


class Gauge(Counter):
    kind = "gauge"

    def set(self, value: float, **labels):
        with self._lock:
            self._v[_key(labels)] = float(value)

    def dec(self, amount: float = 1.0, **labels):
        self.inc(-amount, **labels)


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, name, help_, buckets=DEFAULT_BUCKETS):
        super().__init__(name, help_)
        self.buckets = tuple(sorted(buckets))
        self._data: dict[tuple, dict] = {}

    def observe(self, value: float, **labels):
        k = _key(labels)
        idx = bisect.bisect_left(self.buckets, value)  # first bucket with bound >= value
        with self._lock:
            d = self._data.setdefault(k, {"counts": [0] * (len(self.buckets) + 1), "sum": 0.0, "n": 0})
            d["counts"][idx] += 1
            d["sum"] += value
            d["n"] += 1

    def clear(self):
        with self._lock:
            self._data.clear()

    def quantile(self, q: float, **labels):
        """Estimated quantile (linear interpolation inside the bucket); None when empty."""
        with self._lock:
            d = self._data.get(_key(labels))
            if not d or d["n"] == 0:
                return None
            counts, n = list(d["counts"]), d["n"]
        return self._quantile(counts, n, q)

    def _quantile(self, counts, n, q):
        target, cum = q * n, 0
        for i, c in enumerate(counts):
            if c and cum + c >= target:
                if i >= len(self.buckets):          # +Inf bucket: best we can say is the last bound
                    return self.buckets[-1]
                lo = self.buckets[i - 1] if i else 0.0
                hi = self.buckets[i]
                return lo + (hi - lo) * ((target - cum) / c)
            cum += c
        return self.buckets[-1]

    def render(self) -> list[str]:
        with self._lock:
            items = sorted((k, list(d["counts"]), d["sum"], d["n"]) for k, d in self._data.items())
        lines = self._header()
        for k, counts, total, n in items:
            cum = 0
            for bound, c in zip(self.buckets, counts):
                cum += c
                lines.append(f"{self.name}_bucket{_fmt(list(k) + [('le', f'{bound:g}')])} {cum}")
            lines.append(f"{self.name}_bucket{_fmt(list(k) + [('le', '+Inf')])} {n}")
            lines.append(f"{self.name}_sum{_fmt(k)} {total:.6f}")
            lines.append(f"{self.name}_count{_fmt(k)} {n}")
        return lines

    def snapshot(self):
        with self._lock:
            items = sorted((k, list(d["counts"]), d["sum"], d["n"]) for k, d in self._data.items())
        out = []
        for k, counts, total, n in items:
            out.append({"labels": dict(k), "count": n, "sum": round(total, 6),
                        "avg_ms": round(total / n * 1000, 2) if n else None,
                        **{f"p{int(q * 100)}_ms": round(self._quantile(counts, n, q) * 1000, 2)
                           for q in (0.5, 0.95, 0.99)}})
        return out


class Registry:
    def __init__(self):
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _get(self, cls, name, help_, **kw):
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = cls(name, help_, **kw)
            return self._metrics[name]

    def counter(self, name, help_):
        return self._get(Counter, name, help_)

    def gauge(self, name, help_):
        return self._get(Gauge, name, help_)

    def histogram(self, name, help_, buckets=DEFAULT_BUCKETS):
        return self._get(Histogram, name, help_, buckets=buckets)

    def render(self) -> str:
        with self._lock:
            metrics = list(self._metrics.values())
        return "\n".join(line for m in metrics for line in m.render()) + "\n"

    def snapshot(self) -> dict:
        with self._lock:
            metrics = list(self._metrics.items())
        return {name: m.snapshot() for name, m in metrics}

    def clear(self):
        with self._lock:
            metrics = list(self._metrics.values())
        for m in metrics:
            m.clear()


REGISTRY = Registry()

HTTP_REQUESTS = REGISTRY.counter("healthai_http_requests_total", "HTTP requests by route and status.")
HTTP_LATENCY = REGISTRY.histogram("healthai_http_request_duration_seconds", "HTTP request latency by route.")
HTTP_INFLIGHT = REGISTRY.gauge("healthai_http_requests_in_flight", "Requests currently being served.")
CACHE_LOOKUPS = REGISTRY.counter("healthai_cache_lookups_total", "Cache lookups by kind and result (l1, l2, miss).")
CACHE_DB_EVENTS = REGISTRY.counter("healthai_cache_db_events_total",
                                   "Postgres cache events: write_ok, error, skipped_open_circuit, dropped_write.")
CACHE_BREAKER_STATE = REGISTRY.gauge("healthai_cache_db_breaker_state", "Circuit breaker: 0 closed, 1 half-open, 2 open.")
CACHE_WRITE_QUEUE = REGISTRY.gauge("healthai_cache_write_queue_depth", "Pending background cache writes.")
EXTERNAL_CALLS = REGISTRY.counter("healthai_external_calls_total",
                                  "Calls to external AI providers by provider and outcome.")
EXTERNAL_LATENCY = REGISTRY.histogram("healthai_external_call_duration_seconds",
                                      "External AI provider call latency.",
                                      buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0))
