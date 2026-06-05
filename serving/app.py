"""
SensorOps FastAPI serving application.

Endpoints:
  GET  /health                   — liveness + model status
  POST /api/v1/predict           — score a single sensor reading
  POST /api/v1/predict/batch     — score up to 1000 readings at once
  POST /api/v1/model/reload      — hot-reload model from MLflow / disk
  GET  /api/v1/model/info        — current model metadata

Run locally:
  uvicorn serving.app:app --reload --port 8000

Run via Docker:
  docker compose up serving
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from llm.api import router as llm_router
from serving.feature_pipeline import batch_to_matrix, reading_to_vector
from serving.model_loader import get_model, reload_model
from serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictResult,
    _score_to_severity,
)

API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Lifespan — warm up model on startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[app] warming up model...")
    get_model()
    print(f"[app] model ready: {get_model().model_name}")
    yield
    print("[app] shutting down")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------


app = FastAPI(
    title="SensorOps Anomaly Detection API",
    description=(
        "Real-time anomaly scoring for industrial sensor telemetry. "
        "Part of the SensorOps Industrie 4.0 platform."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(llm_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Liveness and model readiness check",
)
def health() -> HealthResponse:
    try:
        model = get_model()
        model_loaded = True
        model_name = model.model_name
    except Exception:
        model_loaded = False
        model_name = "none"

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=model_name,
        version=API_VERSION,
    )


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/predict",
    response_model=PredictResponse,
    tags=["inference"],
    summary="Score a single sensor reading",
)
def predict(request: PredictRequest) -> PredictResponse:
    """
    Returns an anomaly score [0,1] and severity level for one sensor snapshot.

    - **anomaly_score**: 0 = normal, 1 = maximally anomalous
    - **is_anomaly**: True if score >= model threshold (default 0.6)
    - **severity**: LOW / MEDIUM / HIGH / CRITICAL
    """
    model = get_model()

    try:
        X = reading_to_vector(request.reading)
        score = float(model.score(X)[0])
        is_anomaly = bool(model.predict(X)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Scoring failed: {exc}",
        )

    result = PredictResult(
        machine_id=request.reading.machine_id,
        anomaly_score=round(score, 4),
        is_anomaly=is_anomaly,
        severity=_score_to_severity(score),
        threshold=model.threshold,
    )
    return PredictResponse(result=result, model_name=model.model_name)


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictResponse,
    tags=["inference"],
    summary="Score up to 1000 sensor readings in one call",
)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Scores a batch of readings in a single forward pass — much more efficient
    than calling /predict N times.
    """
    model = get_model()

    try:
        X = batch_to_matrix(request.readings)
        scores = model.score(X)
        predictions = model.predict(X)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch scoring failed: {exc}",
        )

    results = [
        PredictResult(
            machine_id=reading.machine_id,
            anomaly_score=round(float(score), 4),
            is_anomaly=bool(pred),
            severity=_score_to_severity(float(score)),
            threshold=model.threshold,
        )
        for reading, score, pred in zip(request.readings, scores, predictions)
    ]

    return BatchPredictResponse(
        results=results,
        total=len(results),
        anomaly_count=int(sum(r.is_anomaly for r in results)),
        model_name=model.model_name,
    )


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/model/reload",
    tags=["ops"],
    summary="Hot-reload the model from MLflow or disk",
)
def model_reload() -> dict:
    """
    Reloads the model without restarting the server.
    Useful after promoting a new version to Production in MLflow.
    """
    model = reload_model()
    return {
        "status": "reloaded",
        "model_name": model.model_name,
        "threshold": model.threshold,
    }


@app.get(
    "/api/v1/model/info",
    tags=["ops"],
    summary="Current model metadata and hyperparameters",
)
def model_info() -> dict:
    model = get_model()
    return {
        "model_name": model.model_name,
        "threshold": model.threshold,
        "params": model.get_params(),
        "is_fitted": model._is_fitted,
    }
