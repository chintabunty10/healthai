from app.metrics import Counter, Gauge, Histogram, Registry


def test_counter_labels_are_independent():
    c = Counter("c_total", "help")
    c.inc(route="/a"); c.inc(2, route="/a"); c.inc(route="/b")
    assert c.value(route="/a") == 3 and c.value(route="/b") == 1 and c.value(route="/zzz") == 0


def test_gauge_set_inc_dec():
    g = Gauge("g", "help")
    g.set(5); g.inc(); g.dec(2)
    assert g.value() == 4


def test_histogram_quantiles_are_close_to_truth():
    h = Histogram("h_seconds", "help", buckets=(0.01, 0.1, 1.0))
    for _ in range(90):
        h.observe(0.005)            # fast requests
    for _ in range(10):
        h.observe(0.5)              # slow tail
    assert h.quantile(0.5) <= 0.01
    assert 0.1 < h.quantile(0.99) <= 1.0
    assert h.quantile(0.5, route="never-seen") is None


def test_histogram_overflow_bucket_reports_last_bound():
    h = Histogram("h2", "help", buckets=(0.1, 1.0))
    h.observe(50)
    assert h.quantile(0.99) == 1.0


def test_prometheus_text_format():
    r = Registry()
    r.counter("req_total", "Requests.").inc(route='/a"b', status="200")
    h = r.histogram("lat_seconds", "Latency.", buckets=(0.1, 1.0))
    h.observe(0.05); h.observe(0.5)
    text = r.render()
    assert "# TYPE req_total counter" in text
    assert 'req_total{route="/a\\"b",status="200"} 1' in text      # label value escaped
    assert 'lat_seconds_bucket{le="0.1"} 1' in text
    assert 'lat_seconds_bucket{le="1"} 2' in text
    assert 'lat_seconds_bucket{le="+Inf"} 2' in text
    assert "lat_seconds_count 2" in text


def test_snapshot_exposes_percentiles_and_registry_is_idempotent():
    r = Registry()
    h = r.histogram("x_seconds", "x")
    assert r.histogram("x_seconds", "x") is h
    h.observe(0.02, route="/api/predict")
    snap = r.snapshot()["x_seconds"][0]
    assert snap["count"] == 1 and snap["labels"] == {"route": "/api/predict"} and "p95_ms" in snap


def test_metrics_are_thread_safe():
    import threading
    c = Counter("t_total", "t")
    ts = [threading.Thread(target=lambda: [c.inc() for _ in range(2000)]) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert c.value() == 16000
