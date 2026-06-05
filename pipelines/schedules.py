"""
Dagster schedules for SensorOps.

Schedules:
  hourly_pipeline   — run the full pipeline every hour (production cadence)
  daily_pipeline    — run once a day at 06:00 UTC (lighter batch cadence)
"""

from dagster import ScheduleDefinition

from pipelines.jobs import sensorops_full_pipeline

hourly_pipeline_schedule = ScheduleDefinition(
    job=sensorops_full_pipeline,
    cron_schedule="0 * * * *",  # every hour on the hour
    name="hourly_pipeline",
    description="Run the full SensorOps pipeline every hour.",
)

daily_pipeline_schedule = ScheduleDefinition(
    job=sensorops_full_pipeline,
    cron_schedule="0 6 * * *",  # 06:00 UTC daily
    name="daily_pipeline",
    description="Run the full SensorOps pipeline once a day at 06:00 UTC.",
)
