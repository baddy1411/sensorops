"""
Dagster sensors for SensorOps.

Sensors:
  new_csv_sensor — watches data/raw/ for a new/updated CSV file and
                   triggers sensorops_full_pipeline when one appears.

In production this would watch an S3 bucket or Kafka consumer lag.
For local dev it polls the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

from dagster import RunRequest, SensorEvaluationContext, SkipReason, sensor

from pipelines.jobs import sensorops_full_pipeline

_WATCH_PATH = Path("data/raw/ai4i2020.csv")
_CURSOR_KEY = "last_mtime"


@sensor(
    job=sensorops_full_pipeline,
    name="new_csv_sensor",
    description="Triggers full pipeline when the AI4I CSV file is updated.",
    minimum_interval_seconds=60,
)
def new_csv_sensor(context: SensorEvaluationContext):
    """
    Poll the CSV file's mtime. If it changed since last run, trigger the pipeline.
    """
    if not _WATCH_PATH.exists():
        yield SkipReason(f"Dataset not found at {_WATCH_PATH} — waiting for file.")
        return

    current_mtime = str(os.path.getmtime(_WATCH_PATH))
    last_mtime = context.cursor

    if last_mtime == current_mtime:
        yield SkipReason("CSV file unchanged since last check.")
        return

    context.update_cursor(current_mtime)
    yield RunRequest(
        run_key=current_mtime,
        run_config={},
        tags={"trigger": "csv_updated", "mtime": current_mtime},
    )
