"""Structured logging with per-request correlation.

* JSON lines in production (easy to ship to any log stack), readable text for local dev.
* `request_id` is attached to every record emitted while handling a request.
* Privacy: we log counts, cache outcomes and timings - NEVER symptom text, images or keys.
"""
from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
request_ctx_var: ContextVar[dict | None] = ContextVar("request_ctx", default=None)

_STANDARD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "taskName"}


def note(**fields) -> None:
    """Attach fields (e.g. cache='hit') to the access-log line of the request being handled."""
    ctx = request_ctx_var.get()
    if ctx is not None:
        ctx.update(fields)


def _extras(record: logging.LogRecord) -> dict:
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            **_extras(record),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = " ".join(f"{k}={v}" for k, v in _extras(record).items())
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} [{request_id_var.get()}] {record.getMessage()}"
        out = f"{base} {extras}".rstrip()
        return out + ("\n" + self.formatException(record.exc_info) if record.exc_info else "")


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # we emit our own access line
    for noisy in ("httpx", "httpcore", "openai", "google_genai", "psycopg.pool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
