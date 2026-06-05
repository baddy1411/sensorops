"""
Dagster jobs for SensorOps.

Jobs:
  sensorops_full_pipeline  — runs all 4 assets in sequence (ingest → score → alert)
  ingest_only              — just raw_events + feature_matrix (useful for debugging)
  score_and_alert          — feature_matrix → anomaly_scores → alerts (skips re-ingest)
"""

from dagster import AssetSelection, define_asset_job

sensorops_full_pipeline = define_asset_job(
    name="sensorops_full_pipeline",
    description="Full pipeline: ingest → feature engineering → anomaly scoring → alerts",
    selection=AssetSelection.groups("ingest", "features", "scoring", "alerting"),
)

ingest_only = define_asset_job(
    name="ingest_only",
    description="Ingest and feature engineering only — no model scoring.",
    selection=AssetSelection.groups("ingest", "features"),
)

score_and_alert = define_asset_job(
    name="score_and_alert",
    description="Score pre-ingested features and emit alerts.",
    selection=AssetSelection.groups("scoring", "alerting"),
)
