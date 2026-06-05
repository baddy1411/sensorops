"""
Request / response schemas for the SensorOps serving API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Predict request
# ---------------------------------------------------------------------------


class SensorReading(BaseModel):
    """One sensor snapshot sent for scoring."""

    machine_id: str = Field(..., examples=["M14860"])
    air_temperature_k: float = Field(..., ge=290.0, le=320.0, examples=[298.1])
    process_temperature_k: float = Field(..., ge=300.0, le=330.0, examples=[308.6])
    rotational_speed_rpm: float = Field(..., ge=500.0, le=3500.0, examples=[1551.0])
    torque_nm: float = Field(..., ge=1.0, le=100.0, examples=[42.8])
    tool_wear_min: float = Field(..., ge=0.0, le=300.0, examples=[0.0])
    vibration_ms2: float = Field(..., examples=[0.15])


class PredictRequest(BaseModel):
    reading: SensorReading


class BatchPredictRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1, max_length=1000)


# ---------------------------------------------------------------------------
# Predict response
# ---------------------------------------------------------------------------

SeverityLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _score_to_severity(score: float) -> SeverityLevel:
    if score >= 0.90:
        return "CRITICAL"
    if score >= 0.75:
        return "HIGH"
    if score >= 0.60:
        return "MEDIUM"
    return "LOW"


class PredictResult(BaseModel):
    machine_id: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    severity: SeverityLevel
    threshold: float


class PredictResponse(BaseModel):
    result: PredictResult
    model_name: str
    api_version: str = "v1"


class BatchPredictResponse(BaseModel):
    results: list[PredictResult]
    total: int
    anomaly_count: int
    model_name: str
    api_version: str = "v1"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str
    version: str
