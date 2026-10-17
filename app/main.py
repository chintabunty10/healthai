"""FastAPI app: /api/predict (text), /api/predict/image (photos + optional text), /api/explain."""
from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .cache import LRUCache, PostgresStore, TieredCache
from .config import get_settings
from .llm import Explainer
from .model import DISCLAIMER, SymptomModel
from .schemas import ExplainRequest, ExplainResponse, PredictRequest, PredictResponse
from .vision import MAX_IMAGE_BYTES, MAX_IMAGES, ImageAnalyzer, ImageError, load_image

log = logging.getLogger("healthai")
STATIC = Path(__file__).parent / "static"


def build_state(model=None, cache=None, explainer=None, vision=None) -> dict:
    s = get_settings()
    model = model or SymptomModel()
    return {
        "settings": s,
        "model": model,
        "cache": cache or TieredCache(LRUCache(s.cache_max_items), PostgresStore(s.database_url)),
        "explainer": explainer or Explainer(s.groq_api_key, s.groq_base_url, s.groq_model),
        "vision": vision or ImageAnalyzer(s.gemini_api_key, s.gemini_model, model.preprocessor),
    }


def create_app(**overrides) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.h = build_state(**overrides)
        yield

    app = FastAPI(title="HealthAI Symptom Checker", lifespan=lifespan)

    def respond(matched: list[str], unmatched: list[str], **extra) -> PredictResponse:
        """Shared tail of both predict endpoints: cache lookup -> model -> response."""
        h = app.state.h
        key = f"predict:{'|'.join(matched)}"
        cached = h["cache"].get(key)
        was_cached = cached is not None
        if was_cached:
            conditions = cached
        else:
            vector = h["model"].preprocessor.vectorize(matched)
            conds = h["model"].predict(vector, h["settings"].top_k, h["settings"].min_probability)
            conditions = [c.model_dump() for c in conds]
            h["cache"].set(key, conditions)
        return PredictResponse(matched_symptoms=matched, unmatched_symptoms=unmatched,
                               conditions=conditions, cached=was_cached, disclaimer=DISCLAIMER,
                               urgent_warning=h["model"].urgent(matched), **extra)

    def preprocess_text(text: str):
        try:
            return app.state.h["model"].preprocessor.process(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/health")
    def health():
        h = app.state.h
        return {"status": "ok", "cache": h["cache"].memory.stats(),
                "postgres_cache": bool(h["cache"].store and h["cache"].store.enabled),
                "llm_configured": h["explainer"].client is not None,
                "vision_configured": h["vision"].enabled}

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        pre = preprocess_text(req.symptoms)
        if not pre.matched:
            raise HTTPException(status_code=422, detail={
                "message": "None of those symptoms were recognised. Try common terms like 'fever, cough'.",
                "unmatched": pre.unmatched})
        return respond(pre.matched, pre.unmatched)

    @app.post("/api/predict/image", response_model=PredictResponse)
    async def predict_image(images: list[UploadFile] = File(...), symptoms: str = Form("", max_length=500)):
        h = app.state.h
        if not h["vision"].enabled:
            raise HTTPException(status_code=503, detail="Image analysis is not configured on this server.")
        if not 1 <= len(images) <= MAX_IMAGES:
            raise HTTPException(status_code=422, detail=f"Upload between 1 and {MAX_IMAGES} images.")
        raws = [await f.read(MAX_IMAGE_BYTES + 1) for f in images]
        pre = preprocess_text(symptoms)

        digest = hashlib.sha256("".join(sorted(hashlib.sha256(r).hexdigest() for r in raws)).encode()).hexdigest()
        vkey = f"vision:{digest}"
        findings = h["cache"].get(vkey)
        warnings: list[str] = []
        if findings is None:
            try:
                pil = [load_image(r) for r in raws]
            except ImageError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            try:
                result = await run_in_threadpool(h["vision"].analyze, pil)
                findings = {"symptoms": result.symptoms, "description": result.description}
                h["cache"].set(vkey, findings)
            except Exception as exc:  # noqa: BLE001  (network, quota, bad model output)
                log.warning("Vision analysis failed: %s", type(exc).__name__)
                if not pre.matched:
                    raise HTTPException(status_code=502, detail="Image analysis failed. Please try again or describe your symptoms in text.")
                findings = {"symptoms": [], "description": ""}
                warnings.append("Image analysis was unavailable; results use your typed symptoms only.")

        matched = sorted(set(pre.matched) | set(findings["symptoms"]))
        if not matched:
            raise HTTPException(status_code=422, detail={
                "message": "No recognisable symptoms were found in the image. Please add symptoms in text.",
                "unmatched": pre.unmatched})
        return respond(matched, pre.unmatched, image_description=findings["description"] or None,
                       image_symptoms=findings["symptoms"], warnings=warnings)

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
