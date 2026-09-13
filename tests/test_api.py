"""Integration and endpoint test suite for Banking77 FastAPI inference service."""

from fastapi.testclient import TestClient
import pytest

from src.api.app import app
from src.inference import Banking77Predictor


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan context (loads predictor once)."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Verify GET /health returns 200 and accurate service/model metadata."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "distilbert" in data["model_name"].lower()
    assert data["device"] in ["cuda", "cpu"]
    assert data["model_loaded"] is True


def test_root_endpoint(client: TestClient):
    """Verify root GET / endpoint serves the frontend HTML interface."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Banking77 Intent Intelligence" in response.text


def test_api_info_endpoint(client: TestClient):
    """Verify GET /api endpoint returns JSON navigation links."""
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert "docs_url" in data
    assert "health_url" in data
    assert "predict_url" in data


def test_static_assets_endpoint(client: TestClient):
    """Verify static assets (style.css, app.js, config.js) are properly delivered."""
    css_res = client.get("/style.css")
    assert css_res.status_code == 200
    assert "css" in css_res.headers.get("content-type", "")

    js_res = client.get("/app.js")
    assert js_res.status_code == 200
    assert "javascript" in js_res.headers.get("content-type", "")


def test_valid_prediction(client: TestClient):
    """Verify POST /predict returns correct schema, predicted intent, top-3 candidates, and latency."""
    payload = {"text": "My transfer is still pending"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["text"] == payload["text"]
    assert data["predicted_intent"] == "pending_transfer"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["confidence"] > 0.80  # Confidently classified
    assert data["latency_ms"] > 0.0

    # Top-3 predictions check
    assert "top_predictions" in data
    assert len(data["top_predictions"]) == 3
    probs = [p["probability"] for p in data["top_predictions"]]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert data["top_predictions"][0]["intent"] == "pending_transfer"


def test_valid_banking77_intent_output(client: TestClient):
    """Verify predicted intents belong to the 77 canonical Banking77 classes."""
    predictor: Banking77Predictor = client.app.state.predictor
    valid_intents = set(predictor.label2id.keys())

    queries = [
        "My card was stolen while traveling",
        "Why do I need to verify my identity?",
        "I need a replacement card",
    ]

    for q in queries:
        resp = client.post("/predict", json={"text": q})
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_intent"] in valid_intents
        for item in data["top_predictions"]:
            assert item["intent"] in valid_intents


def test_empty_text_error(client: TestClient):
    """Verify empty query string returns HTTP 422 Unprocessable Entity."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_whitespace_text_error(client: TestClient):
    """Verify whitespace-only query string returns HTTP 422 Unprocessable Entity."""
    response = client.post("/predict", json={"text": "    \n\t  "})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_missing_text_error(client: TestClient):
    """Verify payload missing required 'text' field returns HTTP 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_malformed_request_error(client: TestClient):
    """Verify non-string or malformed payload returns HTTP 422."""
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_top_predictions_probabilities(client: TestClient):
    """Verify candidate probabilities are valid probabilities."""
    response = client.post("/predict", json={"text": "I want to exchange euros to pounds"})
    assert response.status_code == 200
    data = response.json()

    candidates = data["top_predictions"]
    assert len(candidates) == 3

    probs = [c["probability"] for c in candidates]
    # Verify probabilities are non-negative and properly ordered
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs[0] >= probs[1] >= probs[2]
    # Top-1 candidate probability matches confidence
    assert pytest.approx(probs[0], abs=1e-3) == data["confidence"]
