"""
Asset: anomaly_scores

Runs the Isolation Forest model over the feature matrix and attaches a
normalised anomaly score [0, 1] to every event.

Score semantics:
  - Higher score → more anomalous
  - Threshold (default 0.6) separates normal from anomaly
  - Scores are derived from sklearn's decision_function, rescaled to [0, 1]
    via min-max normalisation on this batch

MLflow logging:
  - Logs params, metrics, and registers the trained model artifact
  - Uses the MlflowResource for tracking URI + experiment name

Dagster asset key : anomaly_scores
Input             : feature_matrix (DataFrame)
Output            : data/processed/anomaly_scores.parquet  +  MLflow run
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from pipelines.dagster_compat import asset
from pipelines.resources import MlflowResource

OUTPUT_PATH = Path("data/processed/anomaly_scores.parquet")

# Feature columns fed to the model (no label columns)
MODEL_FEATURES = [
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
    "vibration_rolling_mean",
    "vibration_rolling_std",
    "power_rolling_mean",
]

# Isolation Forest hyperparams
IF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": 0.035,  # ~3.5% failure rate in AI4I
    "random_state": 42,
    "n_jobs": -1,
}

ANOMALY_THRESHOLD = 0.6  # scores above this are flagged


def _scores_to_01(raw_scores: np.ndarray) -> np.ndarray:
    """
    Convert Isolation Forest decision_function output to [0, 1].

    decision_function returns negative scores for anomalies.
    We negate so higher = more anomalous, then min-max scale.
    """
    negated = -raw_scores
    lo, hi = negated.min(), negated.max()
    if hi == lo:
        return np.zeros_like(negated)
    return (negated - lo) / (hi - lo)


@asset(
    group_name="scoring",
    description="Isolation Forest anomaly scores [0,1] for every sensor event.",
)
def anomaly_scores(
    context,
    feature_matrix: pd.DataFrame,
    mlflow_resource: MlflowResource,
) -> pd.DataFrame:
    """
    Train (or retrain) an Isolation Forest on the feature matrix,
    score every event, log the run to MLflow, and return the scored DataFrame.
    """
    mlflow.set_tracking_uri(mlflow_resource.tracking_uri)
    mlflow.set_experiment(mlflow_resource.experiment_name)

    X = feature_matrix[MODEL_FEATURES].values

    with mlflow.start_run(run_name="isolation-forest-batch") as run:
        mlflow.log_params(IF_PARAMS)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("anomaly_threshold", ANOMALY_THRESHOLD)
        mlflow.log_param("features", MODEL_FEATURES)

        # Train
        model = IsolationForest(**IF_PARAMS)
        model.fit(X)

        # Score
        raw_scores = model.decision_function(X)
        scores_01 = _scores_to_01(raw_scores)
        predictions = (scores_01 >= ANOMALY_THRESHOLD).astype(int)

        # Metrics
        n_flagged = int(predictions.sum())
        flag_rate = n_flagged / len(predictions)
        mean_score = float(scores_01.mean())

        mlflow.log_metric("n_flagged", n_flagged)
        mlflow.log_metric("flag_rate", flag_rate)
        mlflow.log_metric("mean_anomaly_score", mean_score)

        # Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            model,
            artifact_path="isolation_forest",
            registered_model_name=mlflow_resource.model_name,
        )

        context.log.info(
            f"anomaly_scores: {n_flagged}/{len(X)} flagged "
            f"({flag_rate:.1%}) | MLflow run {run.info.run_id}"
        )

    # Build output DataFrame
    df = feature_matrix.copy()
    df["anomaly_score"] = scores_01
    df["is_anomaly"] = predictions

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    context.add_output_metadata(
        {
            "row_count": len(df),
            "n_flagged": n_flagged,
            "flag_rate": round(flag_rate, 4),
            "mean_score": round(mean_score, 4),
            "mlflow_run_id": run.info.run_id,
            "output_path": str(OUTPUT_PATH),
        }
    )

    return df
