"""
CSV Replay Adapter — AI4I 2020 → SensorEvent stream.

Two modes:
  1. async generator  (default, no external deps)  — yields SensorEvent objects
  2. Kafka producer   (opt-in via KafkaAdapterConfig) — publishes to a topic

Usage — async generator:
    from data.adapter import CsvReplayAdapter
    adapter = CsvReplayAdapter(csv_path="data/raw/ai4i2020.csv")
    async for event in adapter.stream():
        print(event)

Usage — Kafka (requires aiokafka + running broker):
    adapter = CsvReplayAdapter(
        csv_path="data/raw/ai4i2020.csv",
        kafka=KafkaAdapterConfig(bootstrap_servers="localhost:9092", topic="sensorops.raw"),
    )
    await adapter.run_kafka()
"""

from __future__ import annotations

import asyncio
import csv
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from data.schema import FailureType, SensorEvent
from data.vibration import generate_vibration


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReplayConfig:
    """Controls replay speed and timing."""

    # Inter-event delay in seconds. Set to 0 for max throughput.
    event_interval_s: float = 0.1
    # If True, loop back to the start of the CSV after the last row.
    loop: bool = False
    # Optional RNG seed for reproducible vibration synthesis.
    seed: int | None = 42
    # Simulated start time. Defaults to now.
    start_time: datetime | None = None


@dataclass
class KafkaAdapterConfig:
    """Optional Kafka output configuration."""

    bootstrap_servers: str = "localhost:9092"
    topic: str = "sensorops.raw"
    # Number of events to buffer before flushing (aiokafka handles batching internally).
    batch_size: int = 100


# ---------------------------------------------------------------------------
# Column name mapping  (AI4I 2020 CSV → SensorEvent fields)
# ---------------------------------------------------------------------------

_FAILURE_TYPE_COLS: list[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def _parse_failure_type(row: dict[str, str]) -> FailureType:
    """Return the dominant failure subtype, or 'NONE'."""
    for ft in _FAILURE_TYPE_COLS:
        if row.get(ft, "0").strip() == "1":
            return ft  # type: ignore[return-value]
    return "NONE"


def _parse_quality(type_col: str) -> str:
    """AI4I encodes quality as prefix of the 'Type' column: L/M/H."""
    v = type_col.strip().upper()
    if v in ("L", "M", "H"):
        return v
    # Some versions encode as full word
    if v.startswith("L"):
        return "L"
    if v.startswith("M"):
        return "M"
    return "H"


def _row_to_event(
    row: dict[str, str],
    *,
    t: float,
    timestamp: datetime,
    rng: random.Random,
) -> SensorEvent:
    """Convert a raw CSV row dict into a validated SensorEvent."""

    machine_id = row.get("UDI", row.get("Product ID", "UNKNOWN")).strip()
    air_temp = float(row["Air temperature [K]"])
    proc_temp = float(row["Process temperature [K]"])
    rpm = float(row["Rotational speed [rpm]"])
    torque = float(row["Torque [Nm]"])
    wear = float(row["Tool wear [min]"])
    failure = int(row.get("Machine failure", "0")) == 1
    failure_type = _parse_failure_type(row)
    quality = _parse_quality(row.get("Type", "M"))

    vibration = generate_vibration(
        rotational_speed_rpm=rpm,
        tool_wear_min=wear,
        machine_failure=failure,
        t=t,
        rng=rng,
    )

    return SensorEvent(
        event_id=str(uuid.uuid4()),
        machine_id=machine_id,
        timestamp=timestamp,
        air_temperature_k=air_temp,
        process_temperature_k=proc_temp,
        rotational_speed_rpm=rpm,
        torque_nm=torque,
        tool_wear_min=wear,
        vibration_ms2=vibration,
        machine_failure=failure,
        failure_type=failure_type,
        product_quality=quality,  # type: ignore[arg-type]
        source="csv_replay",
    )


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------


class CsvReplayAdapter:
    """
    Replays the AI4I 2020 CSV as a stream of validated SensorEvent objects.

    The adapter adds a synthetic vibration channel to every row and optionally
    publishes events to a Kafka topic.
    """

    def __init__(
        self,
        csv_path: str | Path,
        config: ReplayConfig | None = None,
        kafka: KafkaAdapterConfig | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.config = config or ReplayConfig()
        self.kafka = kafka
        self._rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------
    # Async generator interface
    # ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[SensorEvent]:
        """
        Async generator that yields SensorEvent objects.

        Respects config.event_interval_s between events and config.loop
        to replay the file continuously.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}\n"
                "Download from https://archive.ics.uci.edu/dataset/601 and place in data/raw/"
            )

        start_wall = self.config.start_time or datetime.now(timezone.utc)
        t = 0.0

        while True:
            with open(self.csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    timestamp = datetime(
                        start_wall.year,
                        start_wall.month,
                        start_wall.day,
                        start_wall.hour,
                        start_wall.minute,
                        start_wall.second,
                        tzinfo=timezone.utc,
                    )
                    # Advance simulated clock by event_interval_s
                    from datetime import timedelta
                    timestamp = start_wall + timedelta(seconds=t)

                    try:
                        event = _row_to_event(row, t=t, timestamp=timestamp, rng=self._rng)
                    except Exception as exc:
                        # Log and skip malformed rows rather than crashing the stream
                        print(f"[adapter] skipping malformed row: {exc}")
                        continue

                    yield event

                    t += self.config.event_interval_s
                    if self.config.event_interval_s > 0:
                        await asyncio.sleep(self.config.event_interval_s)

            if not self.config.loop:
                break

    # ------------------------------------------------------------------
    # Kafka interface
    # ------------------------------------------------------------------

    async def run_kafka(self) -> None:
        """
        Publish all events to Kafka.

        Requires aiokafka and a running broker at kafka.bootstrap_servers.
        """
        if self.kafka is None:
            raise ValueError("No KafkaAdapterConfig provided.")

        try:
            from aiokafka import AIOKafkaProducer  # type: ignore[import]
        except ImportError as e:
            raise ImportError("aiokafka is required for Kafka mode: pip install aiokafka") from e

        import json

        producer = AIOKafkaProducer(bootstrap_servers=self.kafka.bootstrap_servers)
        await producer.start()
        published = 0
        try:
            async for event in self.stream():
                payload = event.model_dump_json().encode("utf-8")
                await producer.send_and_wait(self.kafka.topic, payload)
                published += 1
                if published % self.kafka.batch_size == 0:
                    print(f"[adapter] published {published} events to {self.kafka.topic}")
        finally:
            await producer.stop()
            print(f"[adapter] done — {published} total events published")


# ---------------------------------------------------------------------------
# CLI entry point  (python -m data.adapter)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SensorOps CSV Replay Adapter")
    parser.add_argument("--csv", default="data/raw/ai4i2020.csv", help="Path to AI4I 2020 CSV")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds between events (0=max)")
    parser.add_argument("--limit", type=int, default=20, help="Number of events to print then exit")
    parser.add_argument("--loop", action="store_true", help="Loop the CSV indefinitely")
    args = parser.parse_args()

    async def _preview() -> None:
        cfg = ReplayConfig(event_interval_s=args.interval, loop=args.loop)
        adapter = CsvReplayAdapter(csv_path=args.csv, config=cfg)
        count = 0
        async for event in adapter.stream():
            print(event.model_dump_json(indent=2))
            count += 1
            if count >= args.limit:
                break

    asyncio.run(_preview())
