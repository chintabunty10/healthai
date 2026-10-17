"""FastAPI app: /api/predict, /api/explain, /api/health + static UI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .cache import LRUCache, PostgresStore, TieredCache
from .config import get_settings
from .llm import Explainer
from .model import DISCLAIMER, SymptomModel
from .schemas import ExplainRequest, ExplainResponse, PredictRequest, PredictResponse

log = logging.getLogger("healthai")
STATIC = Path(__file__).parent / "static"


def build_state(model=None, cache=None, explainer=None) -> dict:
    s = get_settings()
    return {
        "settings": s,
        "model": model or SymptomModel(),
        "cache": cache or TieredCache(LRUCache(s.cache_max_items), PostgresStore(s.database_url)),
        "explainer": explainer or Explainer(s.groq_api_key, s.groq_base_url, s.groq_model),
    }


def create_app(**overrides) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.h = build_state(**overrides)
        yield

    app = FastAPI(title="HealthAI Symptom Checker", lifespan=lifespan)

    @app.get("/api/health")
    def health():
        h = app.state.h
        return {"status": "ok", "cache": h["cache"].memory.stats(),
                "postgres_cache": bool(h["cache"].store and h["cache"].store.enabled),
                "llm_configured": h["explainer"].client is not None}

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        h = app.state.h
        try:
            pre = h["model"].preprocessor.process(req.symptoms)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not pre.matched:
            raise HTTPException(status_code=422, detail={
                "message": "None of those symptoms were recognised. Try common terms like 'fever, cough'.",
                "unmatched": pre.unmatched})
        key = f"predict:{pre.cache_key}"
        cached = h["cache"].get(key)
        if cached is not None:
            conditions = cached
            was_cached = True
        else:
            conds = h["model"].predict(pre.vector, h["settings"].top_k, h["settings"].min_probability)
            conditions = [c.model_dump() for c in conds]
            h["cache"].set(key, conditions)
            was_cached = False
        return PredictResponse(matched_symptoms=pre.matched, unmatched_symptoms=pre.unmatched,
                               conditions=conditions, cached=was_cached, disclaimer=DISCLAIMER,
                               urgent_warning=h["model"].urgent(pre.matched))

    @app.post("/api/explain", response_model=ExplainResponse)
    async def explain(req: ExplainRequest):
        h = app.state.h
        symptoms = sorted({s.strip().lower() for s in req.symptoms if s.strip()})
        key = f"explain:{req.condition.strip().lower()}:{'|'.join(symptoms)}"
        cached = h["cache"].get(key)
        if cached is not None:
            return ExplainResponse(condition=req.condition, cached=True, **cached)
        fallback = h["model"]._desc.get(req.condition.strip(), "")
        text, source = await run_in_threadpool(h["explainer"].explain, req.condition, symptoms, fallback)
        payload = {"explanation": text, "source": source}
        if source == "llm":  # never cache degraded fallback answers
            h["cache"].set(key, payload)
        return ExplainResponse(condition=req.condition, cached=False, **payload)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
