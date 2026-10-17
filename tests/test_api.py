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
