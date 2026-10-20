# HealthAI Symptom Checker

HealthAI is a small FastAPI web application that turns typed symptoms-and optionally photos of visible signs-into a ranked list of possible conditions. A scikit-learn model performs the ranking; an optional language model can explain a selected condition in plain language.

> **Important:** This project is for educational and informational use only. It does not diagnose, treat, or replace advice from a qualified healthcare professional. The training data is limited and largely synthetic, so results must not be used for medical decisions. Seek urgent medical care for a possible emergency.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Run, test, and retrain](#run-test-and-retrain)
- [How the application works](#how-the-application-works)
- [HTTP API](#http-api)
- [Data and model](#data-and-model)
- [Design decisions and trade-offs](#design-decisions-and-trade-offs)
- [Reliability and observability](#reliability-and-observability)
- [Privacy and security notes](#privacy-and-security-notes)
- [Load test](#load-test)
- [Project layout](#project-layout)
- [Limitations](#limitations)

## Features

- Text symptom entry with normalization, common-language aliases, typo matching, and autocomplete.
- Ranked condition predictions with dataset descriptions, listed precautions, and an urgent-symptom warning.
- Optional photo analysis through Google Gemini. The vision model reports visible signs from the application's symptom vocabulary; the local classifier still ranks conditions.
- Optional plain-language explanations through Groq's OpenAI-compatible API, with a dataset-based fallback.
- In-memory caching by default, with optional Supabase/Postgres caching for expensive explanation and image-analysis results.
- Health and metrics endpoints, structured request logs, and bounded handling of slow or unavailable external services.

The application can start without provider keys or a database. Text prediction remains available; explanations use the fallback when Groq is not configured, image analysis is disabled without Gemini, and caching stays in memory without a database URL.

## Quick start

Commands below assume the current directory is the `healthai` project directory (the directory containing `requirements.txt`). Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m training.train   # optional: regenerate the committed model artifacts
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The interactive API reference is at <http://127.0.0.1:8000/docs>.

If PowerShell does not allow activation, either permit script activation for the current terminal with `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again, or run the venv interpreter directly as `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe -m uvicorn app.main:app --reload`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m training.train   # optional: regenerate the committed model artifacts
uvicorn app.main:app --reload
```

Copying `.env.example` is optional if you do not need external services. Add provider credentials to `.env` only for services you intend to use; never commit `.env`. Run commands from this project directory so Python can resolve `app`, `training`, and the relative `.env` file.

## Configuration

Settings are read from environment variables and `.env`. All entries are optional; defaults are shown below. `.env.example` contains the same settings with short inline guidance.

| Variable | Default | Purpose |
|---|---:|---|
| `GROQ_API_KEY` | empty | Enables LLM-generated explanations. Empty means use the dataset-based fallback. |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible API base URL used by the Groq client. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model used for explanations. |
| `GEMINI_API_KEY` | empty | Enables image understanding. Without it, the image endpoint returns `503`. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model used for image understanding. |
| `DATABASE_URL` | empty | Optional Postgres connection URL (for example, a Supabase pooler URL). Empty means in-memory cache only. |
| `CACHE_MAX_ITEMS` | `1000` | Maximum number of entries in each worker's in-memory LRU cache. |
| `DB_POOL_MIN` / `DB_POOL_MAX` | `1` / `4` | Minimum and maximum Postgres pool connections **per worker**. |
| `DB_POOL_TIMEOUT` | `1.0` | Seconds to wait for a pooled database connection. |
| `DB_STATEMENT_TIMEOUT_MS` | `1500` | Postgres statement timeout in milliseconds. |
| `DB_WRITE_QUEUE_SIZE` | `1000` | Bounded background cache-write queue; excess writes are dropped. |
| `DB_BREAKER_FAILURES` | `3` | Consecutive failures before the database circuit breaker opens. |
| `DB_BREAKER_COOLDOWN` | `30.0` | Seconds before the breaker allows a probe request. |
| `LLM_MAX_CONCURRENCY` | `8` | Maximum simultaneous explanation-provider calls per worker. |
| `VISION_MAX_CONCURRENCY` | `4` | Maximum simultaneous image-provider calls per worker. |
| `EXTERNAL_QUEUE_TIMEOUT` | `2.0` | Seconds a request waits for an external-provider slot before degrading or returning an error. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `LOG_JSON` | `true` | Emit structured JSON logs; set to `false` for readable local logs. |
| `TOP_K` | `5` | Maximum number of conditions returned. |
| `MIN_PROBABILITY` | `0.03` | Minimum probability for additional returned conditions; the highest-ranked result is retained. |

Database caching is optional and best-effort. The app attempts to create its `query_cache` table automatically. If database setup or access fails, the app continues with its in-memory cache. Do not expose a database URL or provider key in source control, logs, screenshots, or public issue reports.

## Run, test, and retrain

### Run the app

The development command in Quick start enables auto-reload. To run multiple worker processes without reload:

```powershell
uvicorn app.main:app --workers 4
```

On macOS/Linux, the same command applies. Each worker has its own model instance, in-memory cache, metrics registry, connection pool, and provider concurrency limits. The total possible database connections are therefore approximately `workers × DB_POOL_MAX`.

### Run tests

```powershell
pytest -q
node tests/ui_smoke.js
```

The Python tests cover preprocessing, API behavior, cache behavior, resilience primitives, metrics/logging, and image validation/parsing. The Node smoke test exercises the real front-end JavaScript with a minimal fake DOM; it is not a substitute for checking the UI in a real browser.

### Retrain the classifier

```powershell
python -m training.train
```

Training reads the CSV files in `data/` and replaces `artifacts/model.joblib`, `artifacts/vocabulary.json`, and `artifacts/metrics.json`. The existing artifacts are committed, so retraining is not required to start the app. The training script uses a fixed random seed for repeatable splits and augmentation.

### Run notebooks

Install the additional notebook dependencies, then launch Jupyter:

```powershell
python -m pip install -r requirements-notebooks.txt
jupyter notebook notebooks/
```

The notebooks document data exploration/preprocessing, an image-understanding demo, and model selection/training. The deployed application loads the checked-in artifacts; notebook execution is not part of application startup.

## How the application works

```text
Browser (plain HTML, CSS, JavaScript)
  ├─ POST /api/predict ────────────────┐
  ├─ POST /api/predict/image ──────────┤
  └─ POST /api/explain ────────────────┤
                                        v
FastAPI (app/main.py)
  ├─ schemas.py       request/response validation
  ├─ preprocessing.py parse → normalize → aliases/fuzzy match → multi-hot vector
  ├─ cache.py         per-process LRU + optional Postgres cache
  ├─ resilience.py    single-flight, provider concurrency limits, circuit breaker support
  ├─ model.py         RandomForest ranking, descriptions, precautions, red-flag warning
  ├─ llm.py           optional Groq explanation + dataset fallback
  ├─ vision.py        optional Gemini image-to-vocabulary symptom extraction
  └─ metrics.py and logging_config.py

training/train.py ──> artifacts/model.joblib, vocabulary.json, metrics.json
```

For a text prediction, the API validates the request, resolves phrases against the trained symptom vocabulary, and rejects input if no symptom is recognized. The recognized symptoms are sorted and deduplicated before being converted to a fixed-length binary vector. The model ranks conditions, and the response includes the original matches, unrecognized phrases, descriptions, precautions, disclaimer, and-when applicable-an urgent warning.

The image route validates and downscales uploaded images before calling Gemini. Gemini output is parsed and restricted to known symptoms; those symptoms are combined with any recognized typed symptoms and passed through the same classifier path. A successful image result can be cached using an image digest; the uploaded image bytes are not stored by this application. If vision analysis fails but typed symptoms were supplied, the request degrades to text-only and returns a warning.

## HTTP API

All routes are served by the FastAPI application. Visit `/docs` while the server is running for the generated OpenAPI reference.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve the browser application. |
| `GET` | `/api/symptoms` | Return the known symptom vocabulary for autocomplete. |
| `POST` | `/api/predict` | Rank conditions from a JSON symptom string. |
| `POST` | `/api/predict/image` | Analyze one to three uploaded images, optionally combined with typed symptoms. |
| `POST` | `/api/explain` | Explain a condition and symptom list using Groq or the fallback. |
| `GET` | `/api/health` | Show app, cache, provider configuration, and in-flight work status. |
| `GET` | `/api/metrics` | Return application metrics as JSON. |
| `GET` | `/metrics` | Return Prometheus text exposition. |

### Text prediction

Request:

```http
POST /api/predict
Content-Type: application/json
```

```json
{"symptoms": "fever, cough and chills"}
```

The `symptoms` field must be a non-blank string of at most 500 characters. The preprocessor accepts at most 20 phrases, each at most 60 characters. A successful response contains `matched_symptoms`, `unmatched_symptoms`, `conditions`, `cached`, `disclaimer`, and `urgent_warning`. Each condition contains a name, probability (0–1), description, and precautions. If no phrase is recognized, the endpoint returns `422`.

### Image prediction

Send `multipart/form-data` to `/api/predict/image`, with one or more `images` fields and an optional `symptoms` text field. Upload between one and three JPEG, PNG, or WEBP images; each file is limited to 5 MB. The server validates the decoded format and dimensions and resizes images to a maximum side of 1024 pixels before sending them to Gemini. Image requests return `503` when Gemini is not configured.

### Explanation

Request:

```http
POST /api/explain
Content-Type: application/json
```

```json
{"condition": "Common Cold", "symptoms": ["cough", "sore throat"]}
```

`condition` is limited to 100 characters; `symptoms` must contain 1–20 entries. The response includes `condition`, `explanation`, `cached`, and `source` (`"llm"` or `"fallback"`). Fallback answers use the dataset description when available and are not persisted in the shared cache.

## Data and model

The application uses the public Disease Symptom Prediction CSV dataset files in `data/`. The current source dataset has 4,920 rows, but those reduce to **304 unique disease/symptom-set pairs**, covering **41 diseases** and **131 symptoms**. These values describe this checked-in dataset and artifacts; they are not a claim about clinical coverage.

The deployed model is a scikit-learn `RandomForestClassifier` with 100 trees and `min_samples_leaf=2`. Its input is a multi-hot vector over the saved symptom vocabulary. `model.py` uses `predict_proba`, sorts the results, and applies `TOP_K` and `MIN_PROBABILITY`. Dataset descriptions and precaution text are loaded separately from the symptom description and precaution CSVs.

Training choices and current held-out metrics:

| Metric | Result |
|---|---:|
| Unique rows after deduplication | 304 |
| Held-out rows | 61 |
| Training rows after subset augmentation | 3,159 |
| Top-1 / top-3 with all held-out symptoms | 1.000 / 1.000 |
| Top-1 / top-3 with at most three symptoms | 0.803 / 0.967 |

The training script deduplicates before the train/test split to avoid identical rows leaking across both sets. It then augments only the training split with 12 randomly sampled symptom subsets per row because users often report fewer symptoms than the dataset lists. Metrics are measured on the held-out split, including a more realistic test that retains at most three symptoms. After evaluation, the script refits the final model on all deduplicated rows (with augmentation) and writes the three artifacts. The perfect full-symptom score reflects the clean, synthetic data and should not be interpreted as real-world accuracy.

The model-selection workbench is in [`notebooks/03_model_selection_and_training.ipynb`](notebooks/03_model_selection_and_training.ipynb); dataset exploration is in [`notebooks/01_data_exploration_preprocessing.ipynb`](notebooks/01_data_exploration_preprocessing.ipynb).

## Design decisions and trade-offs

1. **Deduplicate before evaluating.** The raw rows contain many duplicates. Splitting before deduplication would put near-identical examples in both train and test data and inflate evaluation scores.
2. **Augment with symptom subsets.** Training examples often contain more symptoms than users report. Random subsets make the model encounter shorter symptom lists during training. This does not make the underlying dataset representative of real patients.
3. **Use a compact Random Forest.** A larger/deeper forest performed similarly in experimentation but created a much larger artifact (about 134 MB versus about 3 MB compressed). The shipped 100-tree model keeps the artifact and startup footprint smaller. Single-row inference sets `n_jobs=1` to avoid creating a thread pool per request.
4. **Normalize flexibly, but surface uncertainty.** Exact vocabulary matches are tried first, followed by curated aliases and `difflib` fuzzy matching at a 0.8 cutoff. Fuzzy matching can map an ambiguous typo incorrectly; unresolved phrases are returned to the user rather than silently discarded.
5. **Return ranked candidates, not a diagnosis.** Several conditions can share symptoms. The UI presents probabilities as model scores and includes a medical disclaimer. A configured set of red-flag symptoms adds an urgent-care warning; this is not a complete triage system.
6. **Keep cheap predictions out of the database path.** Predictions are cached in the process-local LRU only (`l2=False`): inference is cheap, while a remote database round trip can cost more. LLM explanations and successful image-analysis findings can use the optional shared Postgres tier.
7. **Make cache persistence best-effort.** Postgres writes and cache hit updates use a bounded write-behind queue. If it fills, writes are dropped because the values can be recomputed. Reads have bounded pool/query timeouts; the circuit breaker limits repeated calls to a failing database. Database failure must not prevent text prediction.
8. **Coalesce identical concurrent work.** Single-flight ensures concurrent requests for the same cold prediction, explanation, or image analysis share one computation/provider call. Provider limiters bound concurrency and shed excess waiting work rather than allowing an unbounded queue.
9. **Separate visible-sign extraction from condition ranking.** Gemini is used only to describe visible signs and select known vocabulary terms. The local classifier produces the condition ranking. This adds external latency, cost, and provider dependency; image analysis is optional.
10. **Use a small browser client.** Plain HTML/CSS/JavaScript avoids a frontend build step. The UI inserts dynamic text as text rather than HTML; a Node smoke test exercises its JavaScript, but does not provide full browser coverage.
11. **Use small built-in metrics.** The app exposes Prometheus-compatible and JSON metrics without a metrics dependency. Metrics are process-local, so they are not aggregated between multiple workers.

## Reliability and observability

- **Health:** `GET /api/health` reports service status, L1/L2 cache state, provider configuration, active provider calls, and coalesced work.
- **Metrics:** `GET /metrics` exposes Prometheus text; `GET /api/metrics` exposes JSON snapshots. Metrics include request counts/latencies, cache outcomes, database breaker/queue state, and external-provider outcomes/latencies.
- **Logs:** each request gets an `X-Request-ID` response header. A valid incoming ID is retained; otherwise a random ID is generated. Logs are JSON by default; set `LOG_JSON=false` for local text logs.
- **Privacy-oriented telemetry:** request logs and metric labels record counts, timings, routes, status codes, and outcomes-not symptom text, image bytes, or API keys.
- **Database degradation:** absent or failing Postgres caching degrades to memory-only behavior. Cache writes are asynchronous and bounded.
- **External service degradation:** the explanation route falls back to dataset text when the provider is unavailable or its limiter is full. If image analysis fails while typed symptoms are available, prediction continues from the typed symptoms and includes a warning.
- **Worker scaling:** each Uvicorn worker owns independent in-process state. Increasing workers increases aggregate capacity, but also multiplies database connection limits and provider concurrency limits.

These controls improve behavior under load; they do not provide authentication, rate limiting, high availability, or a production medical safety review.

## Privacy and security notes

- API keys are read from environment/configuration, not hard-coded. Keep `.env` private.
- The application does not store uploaded image bytes. For image analysis, validated/downscaled images are sent to Google Gemini. For an LLM explanation, the requested condition and symptom list are sent to the configured Groq-compatible provider.
- Cache entries may contain derived explanation text and image-analysis findings; with `DATABASE_URL` configured, eligible entries are persisted in the Postgres `query_cache` table. Prediction entries are process-local only.
- Upload validation restricts file size, count, decoded format, and pixel count before analysis. Image processing runs off the async event loop.
- Request bodies and symptom phrases are not included in the app's request logs. Provider/network errors are logged by error type rather than by logging request content.
- `/metrics` and the API endpoints have no authentication or rate limiting. Do not expose this development/assessment app publicly without adding appropriate access controls, abuse protection, operational monitoring, and a security/privacy review.


## Load test

The Locust scenario in `loadtest/locustfile.py` mixes repeated and varied text predictions with health checks. Run the app in one terminal, then run this command from the project directory in another:

```powershell
locust -f loadtest\locustfile.py --headless -u 100 -r 20 -t 60s --host http://localhost:8000 --csv loadtest\results
```

The script exits non-zero if failures exceed 1% or aggregate p95 latency exceeds 500 ms. Override those thresholds with `LOAD_MAX_FAIL_RATIO` and `LOAD_MAX_P95_MS`. The checked-in local Windows run completed 8,168 requests with no failures, about 140 requests/second, and 190 ms aggregate p95. The app and Locust shared one machine; p99 and maximum latency were much higher during ramp-up. This is an indicative run, not a capacity guarantee. Only `/api/predict` and `/api/health` were exercised; image analysis and LLM explanations were excluded to avoid provider quota and variable network effects. Details are in [`RESULTS.md`](RESULTS.md), and the plot is [`loadtest/loadtest.png`](loadtest/loadtest.png).

## Project layout

```text
healthai/
├── app/
│   ├── main.py                 FastAPI routes, lifespan, and request middleware
│   ├── config.py               Environment and .env settings
│   ├── schemas.py              Pydantic request/response models
│   ├── preprocessing.py        Symptom parsing, normalization, matching, vectorization
│   ├── model.py                Model loading, prediction, dataset knowledge, red flags
│   ├── cache.py                 LRU and optional resilient Postgres cache
│   ├── resilience.py            Single-flight, limiter, and circuit-breaker primitives
│   ├── llm.py                   Optional Groq-compatible explanations
│   ├── vision.py                Image validation and optional Gemini analysis
│   ├── metrics.py               Prometheus-style and JSON metrics
│   ├── logging_config.py        Structured request logging
│   └── static/                  Browser UI (HTML, CSS, JavaScript)
├── artifacts/                   Trained model, vocabulary, and evaluation metrics
├── data/                         Source CSV files
├── docs/                         Demo and technical walkthroughs
├── loadtest/                     Locust test, in-process benchmark, and results
├── notebooks/                    Exploration, image demo, model workbench
├── tests/                        Python tests and browser-JS smoke test
├── training/train.py             Reproducible artifact-generation script
├── requirements.txt              Application and test dependencies
└── requirements-notebooks.txt    Additional notebook dependencies
```

## Limitations

- The model covers only the conditions and symptom patterns in a small, largely synthetic dataset. Performance on real-world symptoms can be substantially worse than the reported held-out scores.
- The model probabilities are not clinically calibrated risk estimates. Fuzzy matching and incomplete symptom descriptions may also affect the ranking.
- Red-flag detection is a fixed list, not comprehensive emergency detection or medical triage.
- External APIs add cost, latency, availability, and data-handling considerations.
- The load-test result is specific to one local run. It does not establish production capacity, multi-instance behavior, or provider-path performance.
- There is no authentication, authorization, rate limiting, or multi-process metrics aggregation.


## AI Usage

- I first designed the application architecture, including the folder structure, data flow from user input to processing and output, required arguments and parameters, input validation, and potential failure scenarios.

- After implementing the core functionality, I tested the application to verify that the main workflow worked as expected. Once the basic testing was successful, I used **Claude** to refine the code, improve readability, strengthen the code structure, and give the application a more professional finish and drafting the documentation.

- I reviewed the suggested changes and made sure they aligned with the intended architecture and functional requirements.