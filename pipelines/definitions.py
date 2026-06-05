"""
Top-level Dagster Definitions object for SensorOps.

This is the single entry point Dagster loads via dagster.yaml / pyproject.toml.
It registers all assets, resources, jobs, schedules, and sensors.
"""

from __future__ import annotations

from dagster import Definitions

from lineage.dagster_resource import LineageResource
from pipelines.assets import alerts, anomaly_scores, feature_matrix, model_comparison, raw_events
from pipelines.jobs import ingest_only, score_and_alert, sensorops_full_pipeline
from pipelines.resources import AlertSinkResource, CsvDatasetResource, MlflowResource
from pipelines.schedules import daily_pipeline_schedule, hourly_pipeline_schedule
from pipelines.sensors import new_csv_sensor

# All assets as a flat list (Dagster resolves the dependency graph from type hints)
_ALL_ASSETS = [raw_events, feature_matrix, anomaly_scores, alerts, model_comparison]

defs = Definitions(
    assets=_ALL_ASSETS,
    resources={
        "csv_dataset": CsvDatasetResource(
            csv_path="data/raw/ai4i2020.csv",
            seed=42,
            batch_size=0,  # 0 = process full CSV
        ),
        "mlflow_resource": MlflowResource(
            tracking_uri="http://localhost:5000",
            experiment_name="sensorops-anomaly-detection",
            model_name="sensorops-isolation-forest",
        ),
        "alert_sink": AlertSinkResource(
            sink_type="file",
            output_path="data/processed/alerts.jsonl",
        ),
        "lineage": LineageResource(
            openlineage_url="",
            audit_log_path="data/processed/audit.jsonl",
        ),
    },
    jobs=[sensorops_full_pipeline, ingest_only, score_and_alert],
    schedules=[hourly_pipeline_schedule, daily_pipeline_schedule],
    sensors=[new_csv_sensor],
)
