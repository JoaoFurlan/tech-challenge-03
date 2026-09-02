"""Tests for the FastAPI service: /predict, /metrics, /health.

Mocks predict_category and the model-loading step -- no ONNX artifact is
committed to git (models/ is gitignored), so these tests must not depend
on it being present.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main_module.model_module, "load_model", lambda: None)
    monkeypatch.setattr(
        main_module, "predict_category", lambda text: "cardiovascular diseases"
    )
    monkeypatch.setattr(main_module, "has_known_vocabulary", lambda text: True)
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_predict_returns_category_and_urgency(client):
    response = client.post("/predict", json={"text": "a routine case"})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "cardiovascular diseases"
    assert body["urgency"] == "attention"  # "routine" de-escalates urgent -> attention
    assert body["low_confidence"] is False
    assert body["message"] is None


def test_predict_forces_attention_when_low_confidence(client, monkeypatch):
    # Regression test: category defaults to "cardiovascular diseases" (the
    # fixture's mock), which would normally give "urgent". With no real
    # vocabulary signal, neither "urgent" nor "normal" is backed by real
    # evidence -- urgency should be fixed to "attention" regardless of what
    # the raw (unreliable) category prediction would imply, not just
    # floored (see docs/technical-decisions.md -- this is exactly the
    # "stomachache" -> cardiovascular/urgent bug that prompted the fix).
    monkeypatch.setattr(main_module, "has_known_vocabulary", lambda text: False)
    response = client.post("/predict", json={"text": "gibberish input"})
    assert response.status_code == 200
    body = response.json()
    assert body["urgency"] == "attention"
    assert body["low_confidence"] is True
    assert body["message"] == main_module.LOW_CONFIDENCE_MESSAGE


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_metrics_endpoint_returns_prometheus_format(client):
    client.post("/predict", json={"text": "some report text"})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "predict_requests_total" in response.text


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
