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
   ├─ cache.py          LRU (in-memory)  ──miss──▶  Supabase Postgres (persistent)
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
| UI with form + results | `app/static/` |
| Caching | `app/cache.py` (key = sorted, normalized symptom set) |
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
   compression (~3 MB). Inference uses `n_jobs=1` because per-request thread fan-out slows
   single-row predictions under concurrency.
2. **Two-layer cache (LRU + Postgres).** Memory is fastest and gives the speed-up under load; Postgres
   survives restarts. Cost: extra complexity and possible staleness, acceptable because predictions
   are deterministic for a given model. Postgres is best-effort: failures degrade to memory-only.
   Fallback (non-LLM) explanations are intentionally **not** cached so a temporary LLM outage
   doesn't pin poor answers.
3. **Fuzzy matching vs. strictness.** Aliases + `difflib` (cutoff 0.8) forgive typos ("fevar") but may
   occasionally mis-map; unrecognised phrases are shown to the user rather than silently dropped.
4. **Top-k probabilities, not a diagnosis.** The UI shows ranked matches with percentages, a
   disclaimer, and an urgent warning when red-flag symptoms (e.g. chest pain) are present.
5. **Image understanding via a separate vision LLM, not a vision classifier.** Gemini only *describes visible
   signs*, constrained to our symptom vocabulary; the trained model still ranks conditions, so photos
   never produce a diagnosis directly. Anything outside the vocabulary (including text embedded in an
   image) is discarded. Cost: an extra network call (~seconds), so results are cached by image hash;
   if Gemini fails and text symptoms exist, the app degrades to text-only with a warning.
6. **Plain HTML/JS instead of a SPA framework.** Fewer moving parts and no build step; text is
   inserted with `textContent` (no XSS). Less polish than React, but fits the time budget.

## Security and correctness
- Input validated server-side (length, count, per-symptom length); parameterized SQL only.
- Uploads: max 3 images, 5 MB each, JPEG/PNG/WEBP verified with Pillow (not trusting the MIME type), pixel
  cap against decompression bombs, downscaled before being sent to Gemini; images are never stored.
- Secrets only via environment variables; `.env` is git-ignored, `.env.example` is committed.
- LLM prompt forbids diagnoses and drug doses; output is shown as plain text.

## Limitations
- Dataset is largely synthetic and covers 41 conditions; real-world accuracy will be lower.
- No auth or rate limiting (out of scope per brief); add before any real deployment.

## Testing
`pytest -q` covers preprocessing, image validation and Gemini-output parsing (with a fake client, no network), cache behaviour (LRU eviction, write-through/promotion, DB-down
no-op) and the API (validation, caching, red flags, explain caching/fallback, UI served).

Load test result (fill in after running): `<users=100, duration=60s, median=… ms, p95=… ms, failures=…>`

## AI usage
- **Claude (Anthropic)** helped plan the architecture and drafted most of the code, tests and this README.
- **Groq (Llama model via OpenAI-compatible API)** is used at runtime for explanations; **Google Gemini** for image understanding.
- What I verified or changed afterwards: *(edit this to reflect what you actually did)* ran the
  tests locally, reviewed the dataset quirks (duplicates, label typos), checked the metrics,
  tested the LLM and Supabase connections with my own keys, and adjusted prompts/aliases.
