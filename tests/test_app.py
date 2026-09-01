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
    with TestClient(main_module.app) as test_client:
        yield test_client


def test_predict_returns_category_and_urgency(client):
    response = client.post("/predict", json={"text": "a routine case"})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "cardiovascular diseases"
    assert body["urgency"] == "attention"  # "routine" de-escalates urgent -> attention


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
