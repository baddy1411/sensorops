"""
Canonical event schema for SensorOps.

Every event flowing through the platform — whether from the CSV replay adapter
or a future real Kafka source — is validated against SensorEvent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Failure subtypes from AI4I 2020 dataset
FailureType = Literal["TWF", "HDF", "PWF", "OSF", "RNF", "NONE"]


class SensorEvent(BaseModel):
    """
    One machine telemetry snapshot.

    Fields sourced directly from the AI4I 2020 dataset plus the synthetic
    vibration channel added for richer signal diversity.
    """

    # Identity
    event_id: str = Field(..., description="Unique event UUID")
    machine_id: str = Field(..., description="Machine / unit identifier, e.g. 'M14860'")
    timestamp: datetime = Field(..., description="Event wall-clock time (UTC)")

    # Core AI4I features
    air_temperature_k: float = Field(..., ge=295.0, le=305.0, description="Air temperature [K]")
    process_temperature_k: float = Field(
        ..., ge=305.0, le=315.0, description="Process temperature [K]"
    )
    rotational_speed_rpm: float = Field(
        ..., ge=1168.0, le=2886.0, description="Rotational speed [rpm]"
    )
    torque_nm: float = Field(..., ge=3.8, le=76.6, description="Torque [Nm]")
    tool_wear_min: float = Field(..., ge=0.0, le=253.0, description="Cumulative tool wear [min]")

    # Synthetic vibration channel
    vibration_ms2: float = Field(..., description="Radial vibration [m/s²] — synthetic channel")

    # Labels
    machine_failure: bool = Field(..., description="Any failure occurred")
    failure_type: FailureType = Field("NONE", description="Dominant failure subtype")

    # Metadata
    product_quality: Literal["L", "M", "H"] = Field(..., description="Product quality variant")
    source: Literal["csv_replay", "kafka", "synthetic"] = Field(
        "csv_replay", description="Origin of this event"
    )

    @field_validator("process_temperature_k")
    @classmethod
    def process_temp_above_air(cls, v: float, info) -> float:
        air = info.data.get("air_temperature_k")
        if air is not None and v < air:
            raise ValueError("process_temperature_k must be >= air_temperature_k")
        return v

    model_config = {"json_encoders": {datetime: lambda dt: dt.isoformat()}}
