import pytest
from fastapi.testclient import TestClient

from app.cache import LRUCache, TieredCache
from app.main import create_app
from app.model import SymptomModel


class FakeExplainer:
    client = object()
    def __init__(self, source="llm"):
        self.calls, self.source = 0, source
    def explain(self, condition, symptoms, fallback_text=""):
        self.calls += 1
        return f"About {condition}", self.source


@pytest.fixture(scope="module")
def model():
    return SymptomModel()


@pytest.fixture
def explainer():
    return FakeExplainer()


@pytest.fixture
def client(model, explainer):
    app = create_app(model=model, cache=TieredCache(LRUCache(100)), explainer=explainer)
    with TestClient(app) as c:
        yield c


def test_ui_page_has_form_and_results_area(client):
    html = client.get("/").text
    assert 'id="symptom-form"' in html and 'id="results"' in html


def test_predict_returns_conditions(client):
    r = client.post("/api/predict", json={"symptoms": "itching, skin rash, nodal skin eruptions"})
    assert r.status_code == 200
    body = r.json()
    assert body["conditions"][0]["name"] == "Fungal infection"
    assert body["cached"] is False and body["disclaimer"]


def test_repeat_query_is_served_from_cache_regardless_of_order_and_case(client):
    first = client.post("/api/predict", json={"symptoms": "Cough, fever"}).json()
    again = client.post("/api/predict", json={"symptoms": "FEVER ,cough"}).json()
    assert first["cached"] is False and again["cached"] is True
    assert first["conditions"] == again["conditions"]


def test_red_flag_symptoms_trigger_urgent_warning(client):
    body = client.post("/api/predict", json={"symptoms": "chest pain, breathlessness"}).json()
    assert body["urgent_warning"]


@pytest.mark.parametrize("payload", [{"symptoms": ""}, {"symptoms": "   "}, {"symptoms": "x" * 501}, {}])
def test_invalid_input_rejected(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_unrecognised_symptoms_return_helpful_422(client):
    r = client.post("/api/predict", json={"symptoms": "zzzqqq"})
    assert r.status_code == 422 and "unmatched" in r.json()["detail"]


def test_too_many_symptoms_rejected(client):
    r = client.post("/api/predict", json={"symptoms": ",".join(f"s{i}" for i in range(30))})
    assert r.status_code == 422


def test_explain_is_cached_after_first_llm_call(client, explainer):
    payload = {"condition": "Common Cold", "symptoms": ["cough", "high_fever"]}
    a = client.post("/api/explain", json=payload).json()
    b = client.post("/api/explain", json=payload).json()
    assert a["cached"] is False and b["cached"] is True and explainer.calls == 1


def test_fallback_explanations_are_not_cached(model):
    ex = FakeExplainer(source="fallback")
    app = create_app(model=model, cache=TieredCache(LRUCache(10)), explainer=ex)
    with TestClient(app) as c:
        p = {"condition": "Common Cold", "symptoms": ["cough"]}
        c.post("/api/explain", json=p); c.post("/api/explain", json=p)
    assert ex.calls == 2


def test_health_endpoint(client):
    assert client.get("/api/health").json()["status"] == "ok"


# ---------- image endpoint ----------
import io as _io

from PIL import Image as _Image

from app.vision import ImageFindings


def _png() -> bytes:
    buf = _io.BytesIO()
    _Image.new("RGB", (40, 40), (210, 90, 90)).save(buf, format="PNG")
    return buf.getvalue()


class FakeVision:
    def __init__(self, enabled=True, symptoms=("itching", "skin_rash"), fail=False):
        self.enabled, self.symptoms, self.fail, self.calls = enabled, list(symptoms), fail, 0
    def analyze(self, images):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return ImageFindings(self.symptoms, "Red itchy patches on skin")


def _app(model, vision):
    return create_app(model=model, cache=TieredCache(LRUCache(100)), explainer=FakeExplainer(), vision=vision)


def test_image_endpoint_merges_image_and_text_symptoms(model):
    with TestClient(_app(model, FakeVision())) as c:
        r = c.post("/api/predict/image", data={"symptoms": "nodal skin eruptions"},
                   files=[("images", ("a.png", _png(), "image/png"))])
    body = r.json()
    assert r.status_code == 200
    assert {"itching", "skin_rash", "nodal_skin_eruptions"} <= set(body["matched_symptoms"])
    assert body["image_symptoms"] == ["itching", "skin_rash"] and body["image_description"]
    assert body["conditions"][0]["name"] == "Fungal infection"


def test_image_analysis_is_cached_by_image_hash(model):
    v = FakeVision()
    with TestClient(_app(model, v)) as c:
        for _ in range(2):
            c.post("/api/predict/image", files=[("images", ("a.png", _png(), "image/png"))])
    assert v.calls == 1


def test_image_endpoint_rejects_non_images_and_too_many(model):
    with TestClient(_app(model, FakeVision())) as c:
        bad = c.post("/api/predict/image", files=[("images", ("a.png", b"hello", "image/png"))])
        many = c.post("/api/predict/image", files=[("images", (f"{i}.png", _png(), "image/png")) for i in range(4)])
    assert bad.status_code == 422 and many.status_code == 422


def test_image_endpoint_503_when_vision_not_configured(model):
    with TestClient(_app(model, FakeVision(enabled=False))) as c:
        r = c.post("/api/predict/image", files=[("images", ("a.png", _png(), "image/png"))])
    assert r.status_code == 503


def test_vision_failure_falls_back_to_text_with_warning(model):
    with TestClient(_app(model, FakeVision(fail=True))) as c:
        ok = c.post("/api/predict/image", data={"symptoms": "cough, fever"},
                    files=[("images", ("a.png", _png(), "image/png"))])
        none = c.post("/api/predict/image", files=[("images", ("b.png", _png(), "image/png"))])
    assert ok.status_code == 200 and ok.json()["warnings"]
    assert none.status_code == 502
