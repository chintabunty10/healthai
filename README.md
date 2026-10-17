# HealthAI – Patient Symptom Checker

A small FastAPI web app: a user types symptoms, a pre-trained scikit-learn model returns ranked
possible conditions (optionally from photos too, via Gemini), and an LLM (Groq, OpenAI-compatible API) explains each one on request.
**Informational only – not a diagnosis.**

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then fill in GROQ_API_KEY / GEMINI_API_KEY / DATABASE_URL
python -m training.train                               # optional: artifacts/ is committed, this retrains
uvicorn app.main:app --reload                          # open http://localhost:8000
# production-like (several worker processes, no reload): uvicorn app.main:app --workers 4
pytest -q                                              # run the tests
```

The app runs without any keys: without `GROQ_API_KEY` explanations fall back to the dataset
description; without `GEMINI_API_KEY` photo upload returns a clear 503 and text still works; without
`DATABASE_URL` the cache is in-memory only.

Notebooks (dataset exploration + image demo): `pip install -r requirements-notebooks.txt`, then
`jupyter notebook notebooks/`.

Load test (100 concurrent users):
```bash
locust -f loadtest/locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000 --csv loadtest/results
```

## Architecture

```
Browser (static HTML/JS)
   │  POST /api/predict (text)   POST /api/predict/image (photos [+ text])   POST /api/explain
   ▼                               ▼
FastAPI (app/main.py) ── validation (pydantic) ──────────────┐
   │                                                          │
   ├─ preprocessing.py  parse → normalize → alias/fuzzy match → multi-hot vector
   ├─ cache.py          L1 LRU (memory) + L2 Supabase (pool, write-behind, circuit breaker)
   ├─ resilience.py     circuit breaker · single-flight · bounded concurrency (load shedding)
   ├─ metrics.py / logging_config.py   /metrics, JSON logs with request ids
   ├─ model.py          RandomForest top-k + descriptions/precautions + red-flag check
   ├─ llm.py            Groq chat completion (timeout, retry, safe fallback)
   └─ vision.py         Gemini image understanding (validate → downscale → vocabulary-only symptoms)
training/train.py  →  artifacts/model.joblib + vocabulary.json + metrics.json
```

| Brief requirement | Where |
|---|---|
| Symptoms in → conditions out | `POST /api/predict`, `app/model.py` |
| Preprocessing/normalization | `app/preprocessing.py` (tests in `tests/test_preprocessing.py`) |
| LLM explanations | `POST /api/explain`, `app/llm.py` |
| Image input (extra) | `POST /api/predict/image`, `app/vision.py` |
| Dataset understanding | `notebooks/01_data_exploration_preprocessing.ipynb` |
| Model selection and training workbench | `notebooks/03_model_selection_and_training.ipynb` (CV, model zoo, tuning, calibration, export) |
| UI with form + results | `app/static/` |
| Caching | `app/cache.py` (key = sorted, normalized symptom set) |
| Metrics and logs | `app/metrics.py`, `app/logging_config.py`, middleware in `app/main.py` |
| Concurrency | stateless workers, thread-safe LRU, `loadtest/locustfile.py` |

### Data and model
Dataset: public "Disease Symptom Prediction" CSVs (41 diseases, 131 symptoms). Key facts and decisions:
- The raw file has 4,920 rows but only **304 unique** (disease, symptom-set) rows. I **dedupe before
  splitting**; otherwise duplicates leak into the test set and accuracy is meaningless.
- Users type 2–4 symptoms but training rows have up to 17, so training rows are **augmented with
  random symptom subsets** (12 per row).
- Model: RandomForest (100 trees, `min_samples_leaf=2`). Gives calibrated-ish probabilities via
  `predict_proba`, handles sparse binary features, and the saved artifact is ~3 MB.
- Held-out results (`artifacts/metrics.json`): top-1/top-3 = 1.00 / 1.00 with all symptoms;
  **0.80 / 0.97 when only 3 symptoms are given**. The first number is inflated by the dataset's
  synthetic, very clean nature; the second is the more honest one.

## Trade-offs

1. **Accuracy vs. response time / size.** A bigger forest (300 trees, deep) scored about the same but
   produced a 134 MB artifact (too big for Git and slower to load). I chose 100 trees with
   compression (~3 MB). Inference runs with `n_jobs=1` to avoid per-request thread fan-out (a design
   choice for single-row inference; not benchmarked separately).
2. **Where the cache lives (tiered, and not for everything).** L1 is an in-process LRU (microseconds). L2 is
   Supabase Postgres, used only for *expensive* results (LLM explanations, image analysis). Predictions
   are memory-only: recomputing one takes milliseconds, while a round trip to a remote database costs
   more than that (the Supabase project is in another region). L2 trades extra moving parts for
   persistence and sharing across workers; it is strictly best-effort and can never fail a request.
   Fallback (non-LLM) explanations are intentionally **not** cached so a temporary LLM outage doesn't
   pin poor answers. Each worker process has its own L1, so hit rate is lower with many workers.
3. **Fuzzy matching vs. strictness.** Aliases + `difflib` (cutoff 0.8) forgive typos ("fevar") but may
   occasionally mis-map; unrecognised phrases are shown to the user rather than silently dropped.
4. **Top-k probabilities, not a diagnosis.** The UI shows ranked matches with percentages, a
   disclaimer, and an urgent warning when red-flag symptoms (e.g. chest pain) are present.
5. **Image understanding via a separate vision LLM, not a vision classifier.** Gemini only *describes visible
   signs*, constrained to our symptom vocabulary; the trained model still ranks conditions, so photos
   never produce a diagnosis directly. Anything outside the vocabulary (including text embedded in an
   image) is discarded. Cost: an extra network call (~seconds), so results are cached by image hash;
   if Gemini fails and text symptoms exist, the app degrades to text-only with a warning.
6. **Write-behind may lose cache writes.** Writes to L2 go through a bounded queue to a background thread;
   when the queue is full or the database is down, writes are dropped. Acceptable for a cache (it is
   recomputable), and it means database trouble can never add latency to a request.
7. **Hand-rolled metrics instead of `prometheus_client`.** Same exposition format, zero dependencies and
   fully unit-tested, at the cost of fewer features (no multiprocess aggregation: each worker reports
   its own numbers). `/metrics` is unauthenticated, fine for this exercise but not for production.
8. **Plain HTML/JS instead of a SPA framework.** Fewer moving parts and no build step; text is
   inserted with `textContent` (no XSS). Less polish than React, but fits the time budget.

## Reliability under load

| Risk at ~100 concurrent users | Mitigation (where) |
|---|---|
| New DB connection per cache call; too many connections to Supabase | Bounded **connection pool** (`PostgresBackend`, `DB_POOL_MAX`), `prepare_threshold=None` for pooler compatibility |
| Request waits on slow DB writes | **Write-behind**: writes/hit-counters go to a bounded queue drained by one background thread; full queue drops, never blocks |
| Slow or hanging DB reads | Pool wait timeout + per-query `statement_timeout`; predictions skip L2 entirely |
| Dead DB gets hammered, every request pays its timeout | **Circuit breaker** (opens after 3 failures, probes once after 30 s); cache degrades to memory-only |
| 100 identical cold requests all recompute / all call the LLM | **Single-flight**: identical concurrent work runs once (`ThreadSingleFlight` for predict, `SingleFlight` for LLM/vision) |
| LLM / vision provider slow or rate-limited, threads pile up | **Bounded concurrency + load shedding** (`Limiter`): after `EXTERNAL_QUEUE_TIMEOUT` the request gets the dataset description instead of queueing |
| Image decoding blocks the event loop | Decoding runs in a worker thread |

Run several processes for real parallelism: `uvicorn app.main:app --workers 4` (each has its own pool, so
total DB connections = workers x `DB_POOL_MAX`).

## Observability

* `GET /metrics` – Prometheus text format; `GET /api/metrics` – JSON with p50/p95/p99; `GET /api/health` – cache,
  breaker state, queue depth, in-flight external calls.
* Key series: `healthai_http_requests_total{route,status}`, `healthai_http_request_duration_seconds`,
  `healthai_cache_lookups_total{kind,result=l1|l2|miss}`, `healthai_cache_db_events_total{event}`,
  `healthai_cache_db_breaker_state` (0 closed, 1 half-open, 2 open), `healthai_cache_write_queue_depth`,
  `healthai_external_calls_total{provider,outcome=llm|fallback|busy|ok|error}`.
* Logs are one JSON object per line with a per-request `X-Request-ID` (an incoming id is kept only if it matches
  `[A-Za-z0-9._-]{1,64}`). Example:
  `{"ts":"...","level":"INFO","msg":"request","request_id":"9f2c...","method":"POST","route":"/api/predict","status":200,"duration_ms":3.1,"cache":"hit","n_symptoms":2}`
* **Privacy:** logs and metrics contain counts, timings and outcomes only, never symptom text, images or keys.
  Set `LOG_JSON=false` for readable local logs.

## Measured behaviour

`python -m loadtest.inprocess_bench` runs 100 concurrent callers x 30 requests through the real preprocessing,
cache and model code (no HTTP) against a simulated database. "original" = synchronous DB calls in the request
path, no coalescing; "current" = this design. Two runs on a shared 2-vCPU sandbox (indicative only; compare rows,
not machines; the DB is simulated, not Supabase):

| Database condition | design | throughput (req/s) | p95 (ms) | DB calls |
|---|---|---|---|---|
| healthy (30 ms) | original | ~700 | ~900 | ~1,020 |
| healthy (30 ms) | current | ~1,250 | ~39 | 0 |
| slow (500 ms) | original | ~335 | ~1,210 | ~980 |
| slow (500 ms) | current | ~1,070 | ~62 | 0 |
| down | original | ~690 | ~850 | ~1,060 |
| down | current | ~1,320 | ~37 | 0 |

"current" makes 0 DB calls here because predictions are memory-only by design; the pool, write-behind queue and
breaker protect the explanation/vision path and are covered by unit tests (e.g. 20 writes against a 200 ms
database return in under 0.2 s; a failing DB is called 3 times, not 10). An intermediate version that still read L2
on every prediction had p95 of 600-860 ms, which is why predictions skip it. This does **not** replace an HTTP
load test:

```bash
uvicorn app.main:app --workers 4 &
locust -f loadtest/locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000 --csv loadtest/results
```
The Locust run exits non-zero if the failure ratio exceeds 1% or p95 exceeds 500 ms (`LOAD_MAX_FAIL_RATIO`,
`LOAD_MAX_P95_MS`). Record your result in the line below after running it.

## Security and correctness
- Input validated server-side (length, count, per-symptom length); parameterized SQL only.
- Uploads: max 3 images, 5 MB each, JPEG/PNG/WEBP verified with Pillow (not trusting the MIME type), pixel
  cap against decompression bombs, downscaled before being sent to Gemini; images are never stored.
- Secrets only via environment variables; `.env` is git-ignored, `.env.example` is committed.
- LLM prompt forbids diagnoses and drug doses; output is shown as plain text.

## Limitations
- The benchmark above is in-process with a simulated DB; real HTTP and Supabase latency (the project region matters) will differ.
- Dataset is largely synthetic and covers 41 conditions; real-world accuracy will be lower.
- No auth or rate limiting (out of scope per brief); add before any real deployment.

## Testing
`pytest -q` covers preprocessing, resilience (circuit breaker states, single-flight, load shedding), metrics and log formatting, image validation and Gemini-output parsing (with a fake client, no network), cache behaviour (LRU eviction, write-through/promotion, DB-down
no-op) and the API (validation, caching, red flags, explain caching/fallback, UI served).

Load test result (fill in after running): `<users=100, duration=60s, median=… ms, p95=… ms, failures=…>`

## AI usage
- **Claude (Anthropic)** helped plan the architecture and drafted most of the code, tests and this README.
- **Groq (Llama model via OpenAI-compatible API)** is used at runtime for explanations; **Google Gemini** for image understanding.
- What I verified or changed afterwards: *(edit this to reflect what you actually did)* ran the
  tests locally, reviewed the dataset quirks (duplicates, label typos), checked the metrics,
  tested the LLM and Supabase connections with my own keys, and adjusted prompts/aliases.
