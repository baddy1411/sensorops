"""
Asset: feature_matrix

Engineers features from raw_events for anomaly detection models.

Feature engineering steps:
  1. Core normalisation (StandardScaler on all numeric channels)
  2. Derived thermodynamic features
       - temp_delta        : process_temp - air_temp  (heat transfer proxy)
       - power_proxy       : torque * rotational_speed / 9549  (kW approx)
  3. Vibration energy
       - vibration_energy  : vibration_ms2 ** 2  (proportional to RMS²)
  4. Wear ratio
       - wear_ratio        : tool_wear_min / 253.0  (normalised 0→1)
  5. Rolling statistics (window=10) on vibration and power proxy
       - vibration_rolling_mean, vibration_rolling_std
       - power_rolling_mean

Dagster asset key : feature_matrix
Input             : raw_events (DataFrame)
Output            : data/processed/features.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from dagster import asset
from sklearn.preprocessing import StandardScaler

OUTPUT_PATH = Path("data/processed/features.parquet")

# Columns used as model inputs (everything below gets scaled)
NUMERIC_COLS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_ms2",
]

ROLLING_WINDOW = 10


@asset(
    group_name="features",
    description="Engineered feature matrix ready for anomaly detection models.",
)
def feature_matrix(
    context,
    raw_events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derives rich features from raw sensor telemetry.

    Returns a DataFrame with both the original normalised columns and all
    derived features. The 'machine_failure' and 'failure_type' columns are
    carried through as labels (not fed to unsupervised models, but used
    for evaluation and the alert layer).
    """
    df = raw_events.copy()

    # ── 1. Derived physics features ─────────────────────────────────────────
    df["temp_delta"] = df["process_temperature_k"] - df["air_temperature_k"]

    # Power ≈ torque [Nm] × angular_velocity [rad/s] → kW
    # ω = rpm × 2π / 60
    df["power_proxy_kw"] = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60 / 1000

    df["vibration_energy"] = df["vibration_ms2"] ** 2

    df["wear_ratio"] = df["tool_wear_min"] / 253.0

    # ── 2. Rolling statistics ────────────────────────────────────────────────
    df["vibration_rolling_mean"] = df["vibration_ms2"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df["vibration_rolling_std"] = (
        df["vibration_ms2"].rolling(ROLLING_WINDOW, min_periods=1).std().fillna(0.0)
    )
    df["power_rolling_mean"] = df["power_proxy_kw"].rolling(ROLLING_WINDOW, min_periods=1).mean()

    # ── 3. Normalise numeric columns ─────────────────────────────────────────
    all_numeric = NUMERIC_COLS + [
        "temp_delta",
        "power_proxy_kw",
        "vibration_energy",
        "wear_ratio",
        "vibration_rolling_mean",
        "vibration_rolling_std",
        "power_rolling_mean",
    ]

    scaler = StandardScaler()
    df[all_numeric] = scaler.fit_transform(df[all_numeric])

    # ── 4. Persist ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    context.add_output_metadata(
        {
            "row_count": len(df),
            "feature_columns": all_numeric,
            "rolling_window": ROLLING_WINDOW,
            "output_path": str(OUTPUT_PATH),
        }
    )

    context.log.info(f"feature_matrix: {len(df)} rows, {len(all_numeric)} feature columns")
    return df
