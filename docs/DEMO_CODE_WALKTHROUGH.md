# Demo script · Code walkthrough (≈ 1 min 40 s)

**What the reviewers judge here:** does what you say *match the code on screen*, and do you show you understand *why* it is
built that way. So: show real files, name real functions, give a reason for each choice, and admit one limitation.
Fluency does not matter. Order and accuracy do.

> Read the script aloud once, then change any wording that isn't how you talk. Only say what is true for you.
> Places marked **⟦ ⟧** depend on decisions you make in notebook 03.

---

## Before you record (5 minutes)

1. `pytest -q` is green and `uvicorn app.main:app` is running.
2. Open these tabs in VS Code, in this order, zoomed in (Ctrl +) so the code is readable on video:
   1. `app/main.py`, scrolled to `predict()` and `respond()`
   2. `app/preprocessing.py`, scrolled to `SymptomPreprocessor.resolve` and `process`
   3. `app/model.py`, scrolled to `SymptomModel.predict`
   4. `app/resilience.py`, scrolled to `ThreadSingleFlight`
3. Confirm the names in this script still exist (line numbers drift, names don't):
   ```bash
   grep -nE "def (predict|respond|resolve|process|vectorize|urgent)\b|class (ThreadSingleFlight|TieredCache)" app/*.py
   ```
4. Have the browser ready on `http://localhost:8000/metrics` in case you want to show a cache hit.

---

## Script

### 0:00 – 0:20 · The path of one request
**On screen:** `app/main.py` → `predict()`.

> “The main work happens in one path: a text request comes into `predict()` in `main.py`. It does three things in order:
> clean the input, check the cache, and only then run the model. I’ll walk through those three.”

### 0:20 – 0:55 · Step 1: preprocessing
**On screen:** `app/preprocessing.py` → `SymptomPreprocessor.resolve`, then `process`.

> “First, preprocessing. Users don’t type model features, they type ‘tummy ache, fevar’. `process` splits the text,
> validates the length and count, and then `resolve` maps each phrase to one of the 131 symptoms the model knows. It tries an exact match,
> then a small alias dictionary for everyday words, then fuzzy matching with `difflib` at a cutoff of 0.8 for typos.
> Anything it can’t map is returned as unmatched and shown to the user instead of silently dropped.
> The output is a sorted, unique list, which gives me a canonical cache key, and a fixed-length vector for the model.
> I built it this way so the model only ever sees valid input, and ‘fever, cough’ and ‘Cough, FEVER’ are the same query.”

### 0:55 – 1:25 · Step 2: cache and single-flight
**On screen:** `respond()` in `main.py`, then `ThreadSingleFlight` in `resilience.py`.

> “Second, `respond()` builds the key from the matched symptoms and asks the cache. I pass `l2=False`: predictions stay in memory only,
> because recomputing takes milliseconds and a round trip to a remote database costs more. On a miss, the work goes through
> `ThreadSingleFlight`. If a hundred users send the same cold query at once, one of them computes and the rest wait for that result.
> I added it after a benchmark showed a CPU stampede on cold keys.”

### 1:25 – 1:50 · Step 3: model and safety
**On screen:** `app/model.py` → `SymptomModel.predict`, then `urgent`.

> “Third, `predict` takes `predict_proba` from the saved ⟦RandomForest / Logistic Regression⟧, sorts it and returns the top-k with descriptions
> and precautions. I return a ranked list with probabilities, not one answer, because several diseases share most of their symptoms.
> `urgent` checks a set of red-flag symptoms, such as chest pain, and adds an emergency warning to the response.”

### 1:50 – 2:00 · One honest limitation
> “One limitation: the dataset is largely synthetic and clean, so real-world accuracy will be lower. I report accuracy with three symptoms, not all of them,
> for that reason.”

*(≈ 330 words; at a relaxed pace this lands near 2:00. If you run long, cut the `difflib` sentence or the red-flag sentence.)*

---

## Optional 30-second extension (only if you have time left)
**On screen:** `app/cache.py` → `ResilientStore` and `TieredCache.get`.

> “For the expensive results, LLM explanations and photo analysis, there is a second cache tier in Supabase. It uses a connection pool,
> writes go through a background queue so they never slow a request, and a circuit breaker stops calling the database after three failures.
> If Supabase is down, the app degrades to memory only.”

---

## Claims you may make, and where the proof is

| You say | Where it is in the code | How to prove it live |
|---|---|---|
| Typos and synonyms are understood | `SymptomPreprocessor.resolve` | Type `fevar, tummy ache` in the UI; the “Based on” chips show the mapped symptoms |
| Same query in any order or case is one cache entry | `PreprocessResult.cache_key`, `respond()` | Run “cough, fever” then “FEVER, cough”; the second shows “from cache” |
| Unrecognised terms are reported, not dropped | `unmatched` in `process` | Type `cough, zzz`; the “Not recognised: zzz” line appears |
| Red-flag symptoms trigger a warning | `RED_FLAGS`, `SymptomModel.urgent` | Type `chest pain, breathlessness` |
| Predictions skip the database tier | `l2=False` in `respond()` | `/metrics` shows `healthai_cache_lookups_total{kind="predict",...}` and no `l2` result for predict |
| Identical concurrent cold queries compute once | `ThreadSingleFlight.do` | `tests/test_resilience.py::test_thread_singleflight_computes_once_for_a_burst_of_identical_calls` |
| Request ids and timings are logged without symptom text | `observe()` middleware, `note()` | Show one log line in the terminal: it has `n_symptoms`, never the words |

## Don't say
- “99% / 100% accurate.” Say: top-1 and top-3 with three symptoms, from cross-validation ⟦your numbers⟧.
- “Production ready” or “handles 100 users” unless you ran the Locust test and have the result in the README.
- Anything about the LLM or Gemini behaving a certain way that you didn't see yourself in the live demo.
- “I wrote everything myself” if you used AI tools. Say what the README's AI-usage section says.

## If a reviewer asks (short answers)
- **Why not read predictions from the database too?** “I measured it: an intermediate version that did had p95 of 600–860 ms against 37–62 ms without, in my in-process benchmark with a simulated database. Real Supabase latency would differ.”
- **Why a fuzzy cutoff of 0.8?** “It forgives one or two typos but rarely maps to the wrong symptom. I haven't tuned it systematically; that is a possible improvement.”
- **Why top-k instead of one label?** “Diseases like the hepatitis types overlap heavily in symptoms. Notebook 01 shows the overlap; a single label would sound more certain than the data supports.”
- **What would you do next?** “Handle full sentences and negation (‘no fever’), which the parser doesn’t do yet, and test on real clinical data.”
