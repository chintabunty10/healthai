# Demo script · Technical explanation (architecture, key decisions, trade-offs)

**What the reviewers judge (18% of the score):** how clearly you explain the architecture, the key decisions and the trade-offs,
in a coherent order. Filler words, pauses and accent do not lower the score. So: say *what you chose, what you rejected, why, and what it costs*,
in that order, and point at evidence.

> Edit every line into your own words. Only claim what you actually ran or read. **⟦ ⟧** marks things to fill in from your own results.

---

## Master timeline (target 7:40, hard limit 8:00)

| Time | Segment | Script |
|---|---|---|
| 0:00–0:25 | Intro | this file |
| 0:25–2:45 | Live demo mapped to the requirements | this file |
| 2:45–5:30 | **Architecture, decisions, trade-offs** | this file |
| 5:30–7:10 | Code walkthrough (webcam on, your face visible) | `docs/DEMO_CODE_WALKTHROUGH.md` |
| 7:10–7:40 | Limitations and AI usage | this file |

---

## 0:00 – 0:25 · Intro
> “Hi, I’m ⟦name⟧. This is HealthAI, a symptom checker: you enter symptoms, a trained model returns ranked possible conditions,
> and a language model explains each one. It is general information, not a diagnosis. I’ll show it against the requirements,
> explain how it’s built and what I traded off, then walk through the main code.”

## 0:25 – 2:45 · Live demo (follow the requirement order)
| Do this | Say this | Requirement |
|---|---|---|
| Open the page | “The page shows an input form and a results area from the start.” | UI |
| Type `fever, cough, chills` | “Typed input is cleaned and mapped to known symptoms: you can see what it understood.” | Preprocessing |
| Point at the ranked list | “The model returns a ranked list with match percentages and precautions.” | ML model |
| Click Explain on one card | “Explanations come from an LLM on request, with a plain-text fallback if it’s down.” | LLM |
| Re-run as `cough, fever` | “Same symptoms, different order: served from cache, 2 ms instead of 64.” (use your real numbers) | Caching |
| Type `chest pain, breathlessness` | “Red-flag symptoms show an urgent warning.” | Safety |
| Attach a photo | “Optionally a photo: Gemini describes visible signs, and only symptoms from our list are kept.” | Extra |
| Open `/metrics` or `/api/health` | “Metrics and request-id logs are built in.” | Observability |

---

## 2:45 – 5:30 · Architecture, key decisions, trade-offs (the scored part)

**Show:** the architecture diagram from the README (or draw this):

```
Browser (HTML/JS)
   │ POST /api/predict   /api/predict/image   /api/explain
   ▼
FastAPI ── validation ──────────────────────────────────────────────┐
   ├─ preprocessing   parse → normalize → alias/fuzzy → vector       │ middleware:
   ├─ cache           L1 memory LRU  +  L2 Supabase (pool,           │ request id, timing,
   │                  write-behind queue, circuit breaker)           │ /metrics, JSON logs
   ├─ model           scikit-learn ⟦model⟧ → top-k + red-flag check   │
   ├─ llm             Groq explanations (limits, fallback)           │
   └─ vision          Gemini photo → symptoms from our vocabulary    │
training/ + notebooks/ → artifacts/model.joblib + vocabulary.json ───┘
```

### Script (~2 min 45 s)

**Architecture (25 s)**
> “It’s a single FastAPI service with small modules: preprocessing, model, cache, LLM and vision. Training is separate:
> a script and notebooks produce a model file and a symptom vocabulary that the service loads at startup.
> The same vocabulary is used for typed text and for photos, so there’s one definition of valid input.”

**Decision 1: the model, chosen by evidence (40 s)**
> “I compared ⟦number⟧ algorithms with repeated cross-validation, in a notebook, and chose ⟦model⟧. Two details mattered.
> First, the raw file has 4,920 rows but only 304 unique, so I de-duplicate *before* splitting; a naive split put 100% of test patterns in training.
> Second, users type two to four symptoms, but training rows have up to seventeen, so I augment training rows by randomly dropping symptoms,
> inside each fold. I evaluate on three-symptom inputs: top-3 accuracy is ⟦0.xx⟧. I also checked probability quality, because the UI shows percentages;
> ⟦one sentence about calibration, e.g. ‘the RandomForest was under-confident by ~29 points, Logistic Regression was within 1’⟧.”

**Decision 2: LLM and vision are advisory, not the decision-maker (25 s)**
> “The LLM only explains; it never ranks conditions. Gemini only *describes visible signs*, and its answer is restricted to our symptom list,
> so text hidden in an image can’t inject content. The trained model remains the only thing that produces a ranked list.”

**Decision 3: where the cache lives (35 s)**
> “Caching is two tiers. Memory is fast. Supabase adds persistence, but only for expensive results, LLM explanations and photo analysis.
> Predictions are memory-only, because recomputing takes milliseconds and a round trip to a remote database costs more.
> My first design read the database on every miss, and my benchmark showed p95 around 600 to 860 milliseconds against about 40 to 60 without that read.
> The database tier uses a connection pool, writes go through a background queue so they never delay a request, and a circuit breaker stops calling it after repeated failures.”

**Trade-offs (40 s)**
> “Every choice has a cost. Background writes can be dropped when the queue is full: acceptable for a cache, and it means database trouble can’t slow a user.
> Each worker process has its own memory cache, so hit rate drops with many workers. Fuzzy matching forgives typos but can occasionally mis-map,
> so unrecognised words are shown to the user instead of guessed. Metrics are hand-written, with no extra dependency, but they aren’t aggregated across processes.
> And I return top-k probabilities rather than one diagnosis, because several diseases share most of their symptoms.”

### Evidence you can flash on screen (pick 2–3, don't read them)
- Notebook 03: the model comparison table and the reliability (calibration) plot.
- Notebook 01: the “key findings → decisions” table (duplicates, leakage, overlap).
- README “Measured behaviour” table (in-process benchmark; say it uses a *simulated* database).
- Terminal: `pytest -q` showing ⟦N⟧ tests passing.
- `/metrics` showing `healthai_cache_lookups_total` with `l1` hits.

---

## 5:30 – 7:10 · Code walkthrough
Use `docs/DEMO_CODE_WALKTHROUGH.md`. Webcam on, face visible.

## 7:10 – 7:40 · Limitations and AI usage
> “Limitations: the dataset is largely synthetic and covers 41 conditions, so real-world accuracy will be lower; there’s no authentication or
> rate limiting, which the brief excluded; and the parser handles comma-separated terms but not full sentences or negation yet.
> ⟦Load test: ‘Under a 100-user Locust run, p95 was … ms with … failures.’ — only if you ran it.⟧
> I used Claude to help plan and draft code, tests and documentation. I verified it by ⟦e.g. running the full test suite, reading the dataset and
> choosing the model myself in a notebook, and testing the live Groq, Gemini and Supabase connections⟧, and I changed ⟦what you actually changed⟧.
> The README has the full AI-usage section. Thanks for watching.”

---

## Trade-off cheat sheet (for the deeper questions)

| Decision | Chose | Over | Because | Costs | Evidence |
|---|---|---|---|---|---|
| Model | ⟦model⟧ | RandomForest by default | ⟦CV result, calibration, size⟧ | ⟦ ⟧ | notebook 03 |
| Split | de-duplicate first | random split of raw rows | duplicates leak into test | fewer rows (304) | notebook 01/03 |
| Input realism | symptom-dropout augmentation | train on full rows only | users give 2–4 symptoms | synthetic variants | augmentation table in notebook 03 |
| Cache for predictions | memory only | read/write Supabase too | DB round trip > recompute | lost on restart | benchmark in README |
| Cache for LLM/vision | memory + Supabase | memory only | expensive, worth persisting | more moving parts | `cache.py` |
| DB failures | breaker + background writes | direct synchronous calls | a DB problem must not slow users | dropped cache writes | `tests/test_cache.py` |
| Identical concurrent work | single-flight | let every request compute | avoid CPU/LLM stampede | waiters share one failure | `tests/test_resilience.py` |
| LLM overload | limit + fallback text | queue everything | bounded latency under load | less rich answer when busy | `test_saturated_llm_sheds_load...` |
| Photos | Gemini → vocabulary symptoms | Gemini classifies disease | keeps the trained model as the ranker; safer | extra network call | `app/vision.py` |
| Metrics | hand-written, Prometheus format | `prometheus_client` | zero dependencies, fully tested | no multi-process aggregation | `tests/test_metrics.py` |
| Front end | plain HTML/JS | React SPA | no build step, fits the time budget | less polish | `app/static/` |
