"""FastAPI triage inference service: /predict, /metrics, /health.

/predict is plain `def`, not `async def` — inference is CPU-bound, so
there's no I/O to yield on (see docs/architecture.md). Model is loaded
once at startup via the FastAPI lifespan (not per-request, not at import
time — see app/model.py for why).
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app import model as model_module
from app.model import has_known_vocabulary, predict_category
from app.urgency import floor_urgency, predict_urgency

REQUEST_COUNT = Counter("predict_requests_total", "Total /predict requests", ["status"])
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Latency of /predict requests")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_module.load_model()
    yield


app = FastAPI(title="MedSys Triage API", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Medical report text (laudo médico)")


class PredictResponse(BaseModel):
    category: str
    urgency: str
    low_confidence: bool = Field(
        description=(
            "True when the input shares no vocabulary with the training "
            "data (e.g. very short or colloquial text) -- the category is "
            "then driven by the classifier's structural bias rather than "
            "real evidence; urgency is floored at 'attention' rather than "
            "trusting a possibly-spurious 'normal'. See technical-decisions.md."
        )
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    start = time.perf_counter()
    try:
        category = predict_category(request.text)
        urgency = predict_urgency(category, request.text)
        low_confidence = not has_known_vocabulary(request.text)
        if low_confidence:
            urgency = floor_urgency(urgency, "attention")
    except Exception:
        REQUEST_COUNT.labels(status="error").inc()
        raise
    else:
        REQUEST_COUNT.labels(status="success").inc()
        return PredictResponse(category=category, urgency=urgency, low_confidence=low_confidence)
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
