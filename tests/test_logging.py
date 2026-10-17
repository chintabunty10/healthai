import json
import logging

from app.logging_config import JsonFormatter, TextFormatter, note, request_ctx_var, request_id_var


def _record(msg="hello", **extra):
    r = logging.LogRecord("healthai.test", logging.INFO, __file__, 1, msg, (), None)
    r.__dict__.update(extra)
    return r


def test_json_formatter_includes_request_id_and_extras():
    token = request_id_var.set("req-123")
    try:
        line = JsonFormatter().format(_record("request", status=200, duration_ms=12.5))
    finally:
        request_id_var.reset(token)
    data = json.loads(line)
    assert data["msg"] == "request" and data["request_id"] == "req-123"
    assert data["status"] == 200 and data["duration_ms"] == 12.5
    assert data["level"] == "INFO" and data["ts"].endswith("+00:00")


def test_json_formatter_survives_unserializable_extras_and_exceptions():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = _record("failed", obj=object())
        rec.exc_info = sys.exc_info()
    data = json.loads(JsonFormatter().format(rec))
    assert "ValueError: boom" in data["exc"] and isinstance(data["obj"], str)


def test_text_formatter_is_human_readable():
    out = TextFormatter().format(_record("request", status=200))
    assert "request" in out and "status=200" in out


def test_note_collects_fields_only_inside_a_request_context():
    note(ignored=True)                        # no context -> silently ignored
    ctx = {}
    token = request_ctx_var.set(ctx)
    try:
        note(cache="hit", n_symptoms=2)
    finally:
        request_ctx_var.reset(token)
    assert ctx == {"cache": "hit", "n_symptoms": 2}
