"""
Lightweight feature engineering for real-time inference.

Mirrors the logic in pipelines/assets/features.py but operates on
a single SensorReading (or a small batch) without Dagster or pandas.

Used by the FastAPI app to transform raw API input into the feature
vector the trained model expects.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from serving.schemas import SensorReading

# Must match pipelines/assets/anomaly_scores.py MODEL_FEATURES — order matters
FEATURE_ORDER = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_ms2",
    "temp_delta",
    "power_proxy_kw",
    "vibration_energy",
    "wear_ratio",
    "vibration_rolling_mean",   # single-step inference: same as vibration_ms2
    "vibration_rolling_std",    # single-step inference: 0.0
    "power_rolling_mean",       # single-step inference: same as power_proxy_kw
]


def reading_to_vector(reading: SensorReading) -> np.ndarray:
    """Convert a single SensorReading to a 1-D feature vector."""
    temp_delta = reading.process_temperature_k - reading.air_temperature_k
    power_kw = reading.torque_nm * reading.rotational_speed_rpm * 2 * math.pi / 60 / 1000
    vib_energy = reading.vibration_ms2 ** 2
    wear_ratio = reading.tool_wear_min / 253.0

    return np.array(
        [
            reading.air_temperature_k,
            reading.process_temperature_k,
            reading.rotational_speed_rpm,
            reading.torque_nm,
            reading.tool_wear_min,
            reading.vibration_ms2,
            temp_delta,
            power_kw,
            vib_energy,
            wear_ratio,
            reading.vibration_ms2,   # rolling_mean approximation
            0.0,                      # rolling_std (no history available)
            power_kw,                 # power_rolling_mean approximation
        ],
        dtype=np.float64,
    ).reshape(1, -1)


def batch_to_matrix(readings: list[SensorReading]) -> np.ndarray:
    """Convert a list of readings to a 2-D feature matrix (n, 13)."""
    return np.vstack([reading_to_vector(r) for r in readings])
