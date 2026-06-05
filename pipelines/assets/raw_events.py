"""
Asset: raw_events

Reads the AI4I 2020 CSV through the replay adapter and materialises a
batch of validated SensorEvent records as a Parquet file.

Dagster asset key : raw_events
Output            : data/processed/raw_events.parquet
Metadata emitted  : row_count, failure_count, failure_rate, columns
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
from dagster import asset

from data.adapter import CsvReplayAdapter, ReplayConfig
from data.schema import SensorEvent
from pipelines.resources import CsvDatasetResource

OUTPUT_PATH = Path("data/processed/raw_events.parquet")


@asset(
    group_name="ingest",
    description="Validated SensorEvent batch from the AI4I 2020 CSV replay adapter.",
)
def raw_events(context, csv_dataset: CsvDatasetResource) -> pd.DataFrame:
    """
    Stream the AI4I 2020 CSV through the replay adapter, validate every row
    against the SensorEvent schema, and return a DataFrame.

    Saves to Parquet for downstream assets.
    """
    cfg = ReplayConfig(event_interval_s=0.0, seed=csv_dataset.seed, loop=False)
    adapter = CsvReplayAdapter(csv_path=csv_dataset.resolved_path(), config=cfg)

    events: list[SensorEvent] = asyncio.run(_collect(adapter, csv_dataset.batch_size))

    df = pd.DataFrame([e.model_dump() for e in events])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    failure_count = int(df["machine_failure"].sum())
    failure_rate = failure_count / len(df) if len(df) > 0 else 0.0

    context.add_output_metadata(
        {
            "row_count": len(df),
            "failure_count": failure_count,
            "failure_rate": round(failure_rate, 4),
            "columns": list(df.columns),
            "output_path": str(OUTPUT_PATH),
        }
    )

    context.log.info(
        f"raw_events: {len(df)} rows loaded, "
        f"{failure_count} failures ({failure_rate:.1%})"
    )
    return df


async def _collect(adapter: CsvReplayAdapter, limit: int) -> list[SensorEvent]:
    events: list[SensorEvent] = []
    async for event in adapter.stream():
        events.append(event)
        if limit > 0 and len(events) >= limit:
            break
    return events
