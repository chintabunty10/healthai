"""FastAPI app: /api/predict (text), /api/predict/image (photos + optional text), /api/explain,
plus operational endpoints /api/health, /metrics (Prometheus) and /api/metrics (JSON)."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .cache import LRUCache, TieredCache, make_store
from .config import get_settings
from .llm import Explainer
from .logging_config import configure_logging, note, request_ctx_var, request_id_var
from .metrics import (EXTERNAL_CALLS, EXTERNAL_LATENCY, HTTP_INFLIGHT, HTTP_LATENCY, HTTP_REQUESTS, REGISTRY)
from .model import DISCLAIMER, SymptomModel
from .resilience import Busy, Limiter, SingleFlight, ThreadSingleFlight
from .schemas import ExplainRequest, ExplainResponse, PredictRequest, PredictResponse
from .vision import MAX_IMAGE_BYTES, MAX_IMAGES, ImageAnalyzer, ImageError, load_image

log = logging.getLogger("healthai")
STATIC = Path(__file__).parent / "static"
_RID_OK = re.compile(r"[A-Za-z0-9._-]{1,64}")
QUIET_PATHS = {"/metrics", "/api/health", "/api/metrics"}  # logged at DEBUG to keep INFO useful


def build_state(model=None, cache=None, explainer=None, vision=None, llm_limiter=None, vision_limiter=None) -> dict:
    s = get_settings()
    model = model or SymptomModel()
    if cache is None:
        store = make_store(
            s.database_url, min_size=s.db_pool_min, max_size=s.db_pool_max, pool_timeout=s.db_pool_timeout,
            statement_timeout_ms=s.db_statement_timeout_ms, queue_size=s.db_write_queue_size,
            breaker_failures=s.db_breaker_failures, breaker_cooldown=s.db_breaker_cooldown)
        cache = TieredCache(LRUCache(s.cache_max_items), store)
    return {
        "settings": s,
        "model": model,
        "cache": cache,
        "explainer": explainer or Explainer(s.groq_api_key, s.groq_base_url, s.groq_model),
        "vision": vision or ImageAnalyzer(s.gemini_api_key, s.gemini_model, model.preprocessor),
        "llm_limiter": llm_limiter or Limiter(s.llm_max_concurrency, s.external_queue_timeout),
        "vision_limiter": vision_limiter or Limiter(s.vision_max_concurrency, s.external_queue_timeout),
        "flight": SingleFlight(),                # async work (LLM / vision)
        "predict_flight": ThreadSingleFlight(),  # sync work (model inference) in worker threads
    }


def create_app(**overrides) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = get_settings()
        configure_logging(s.log_level, s.log_json)
        app.state.h = build_state(**overrides)
        h = app.state.h
        log.info("startup", extra={"llm_configured": h["explainer"].client is not None,
                                   "vision_configured": h["vision"].enabled,
                                   "db_cache": bool(h["cache"].store), "cache_max_items": s.cache_max_items})
        yield
        store = h["cache"].store
        if store:
            store.close()  # flush queued writes, close pool
        log.info("shutdown")

    app = FastAPI(title="HealthAI Symptom Checker", lifespan=lifespan)

    # ------------------------------------------------------------------ observability middleware
    @app.middleware("http")
    async def observe(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _RID_OK.fullmatch(incoming) else uuid.uuid4().hex[:16]  # never trust/log raw input
        request_id_var.set(rid)
        ctx: dict = {}
        request_ctx_var.set(ctx)
        start, status = time.perf_counter(), 500
        HTTP_INFLIGHT.inc()
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        except Exception:
            log.exception("unhandled_error")
            raise
        finally:
            HTTP_INFLIGHT.dec()
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            path = getattr(route, "path", None) or "other"   # route template: bounded label cardinality
            if path not in QUIET_PATHS:
                HTTP_REQUESTS.inc(method=request.method, route=path, status=str(status))
                HTTP_LATENCY.observe(elapsed, route=path)
            log.log(logging.DEBUG if path in QUIET_PATHS else logging.INFO, "request",
                    extra={"method": request.method, "route": path, "status": status,
                           "duration_ms": round(elapsed * 1000, 2), **ctx})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse({"detail": "Internal server error", "request_id": request_id_var.get()}, status_code=500)

    # ------------------------------------------------------------------ shared helpers
    def respond(matched: list[str], unmatched: list[str], **extra) -> PredictResponse:
        """Shared tail of both predict endpoints: cache lookup -> model -> response."""
        h = app.state.h
        key = f"predict:{'|'.join(matched)}"
        cached = h["cache"].get(key, l2=False)   # predictions are cheap: memory only, no DB round trip
        was_cached = cached is not None
        if was_cached:
            conditions = cached
        else:
            def compute():
                vector = h["model"].preprocessor.vectorize(matched)
                conds = h["model"].predict(vector, h["settings"].top_k, h["settings"].min_probability)
                out = [c.model_dump() for c in conds]
                h["cache"].set(key, out, l2=False)
                return out
            # a burst of identical cold queries computes once; the rest wait for that result
            conditions = h["predict_flight"].do(key, compute)
        note(cache="hit" if was_cached else "miss", n_symptoms=len(matched))  # counts only, never symptom text
        return PredictResponse(matched_symptoms=matched, unmatched_symptoms=unmatched,
                               conditions=conditions, cached=was_cached, disclaimer=DISCLAIMER,
                               urgent_warning=h["model"].urgent(matched), **extra)

    def preprocess_text(text: str):
        try:
            return app.state.h["model"].preprocessor.process(text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    # ------------------------------------------------------------------ operational endpoints
    @app.get("/api/health")
    def health():
        h = app.state.h
        return {"status": "ok", "cache": h["cache"].status(),
                "llm_configured": h["explainer"].client is not None,
                "vision_configured": h["vision"].enabled,
                "in_flight": {"llm": h["llm_limiter"].active, "vision": h["vision_limiter"].active,
                              "coalesced_work": h["flight"].in_flight}}

    @app.get("/metrics")
    def prometheus_metrics():
        return PlainTextResponse(REGISTRY.render(), media_type="text/plain; version=0.0.4")

    @app.get("/api/metrics")
    def json_metrics():
        return REGISTRY.snapshot()

    # ------------------------------------------------------------------ prediction endpoints
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
        note(images=len(raws))

        digest = hashlib.sha256("".join(sorted(hashlib.sha256(r).hexdigest() for r in raws)).encode()).hexdigest()
        vkey = f"vision:{digest}"
        findings = h["cache"].get(vkey)
        warnings: list[str] = []
        note(image_cache="hit" if findings is not None else "miss")

        async def analyze_once():
            # image decoding is CPU-bound: keep it off the event loop
            pil = await asyncio.to_thread(lambda: [load_image(r) for r in raws])
            t0 = time.perf_counter()
            try:
                result = await h["vision_limiter"].run(h["vision"].analyze, pil)
            except Busy:
                EXTERNAL_CALLS.inc(provider="gemini", outcome="busy")
                raise
            except Exception:
                EXTERNAL_CALLS.inc(provider="gemini", outcome="error")
                raise
            EXTERNAL_CALLS.inc(provider="gemini", outcome="ok")
            EXTERNAL_LATENCY.observe(time.perf_counter() - t0, provider="gemini")
            out = {"symptoms": result.symptoms, "description": result.description}
            h["cache"].set(vkey, out)
            return out

        if findings is None:
            try:
                findings = await h["flight"].do(vkey, analyze_once)   # identical uploads share one Gemini call
            except ImageError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            except Exception as exc:  # noqa: BLE001  (busy, network, quota, bad model output)
                log.warning("vision_unavailable", extra={"error": type(exc).__name__})
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
            note(explain="cache")
            return ExplainResponse(condition=req.condition, cached=True, **cached)
        fallback = h["model"]._desc.get(req.condition.strip(), "")

        async def compute():
            t0 = time.perf_counter()
            try:
                text, source = await h["llm_limiter"].run(h["explainer"].explain, req.condition, symptoms, fallback)
                outcome = source
            except Busy:   # provider saturated: answer from the dataset instead of queueing forever
                text, source, outcome = h["explainer"].fallback(req.condition, fallback), "fallback", "busy"
            EXTERNAL_CALLS.inc(provider="groq", outcome=outcome)
            EXTERNAL_LATENCY.observe(time.perf_counter() - t0, provider="groq")
            payload = {"explanation": text, "source": source}
            if source == "llm":  # never cache degraded fallback answers
                h["cache"].set(key, payload)
            return payload

        payload = await h["flight"].do(key, compute)   # identical concurrent requests share one LLM call
        note(explain=payload["source"])
        return ExplainResponse(condition=req.condition, cached=False, **payload)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
