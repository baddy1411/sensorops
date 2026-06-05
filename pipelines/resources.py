"""
Dagster resources for SensorOps.

Resources are shared, configured dependencies injected into assets and ops.
Defined here so every asset can import from one place.

Resources:
  - CsvDatasetResource    : path + replay config for the AI4I dataset
  - MlflowResource        : MLflow tracking URI + experiment name
  - AlertSinkResource     : where to send alerts (console / file / webhook)
"""

from __future__ import annotations

from pathlib import Path

from dagster import ConfigurableResource, EnvVar
from pydantic import Field


class CsvDatasetResource(ConfigurableResource):
    """Locates the AI4I 2020 CSV and controls replay parameters."""

    csv_path: str = Field(
        default="data/raw/ai4i2020.csv",
        description="Path to the AI4I 2020 CSV file.",
    )
    seed: int = Field(default=42, description="RNG seed for vibration synthesis.")
    batch_size: int = Field(
        default=500,
        description="Number of rows to process per pipeline run (0 = all rows).",
    )

    def resolved_path(self) -> Path:
        return Path(self.csv_path)


class MlflowResource(ConfigurableResource):
    """MLflow experiment tracking configuration."""

    tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI.",
    )
    experiment_name: str = Field(
        default="sensorops-anomaly-detection",
        description="MLflow experiment to log runs under.",
    )
    model_name: str = Field(
        default="sensorops-isolation-forest",
        description="Registered model name in MLflow Model Registry.",
    )


class AlertSinkResource(ConfigurableResource):
    """Configures where anomaly alerts are emitted."""

    sink_type: str = Field(
        default="console",
        description="One of: console, file, webhook.",
    )
    output_path: str = Field(
        default="data/processed/alerts.jsonl",
        description="File path for file sink.",
    )
    webhook_url: str = Field(
        default="",
        description="Webhook URL for webhook sink (e.g. Slack incoming webhook).",
    )
