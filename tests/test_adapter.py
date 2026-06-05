"""
Tests for the CSV replay adapter and vibration generator.

These tests use a synthetic mini-CSV (no real dataset required)
so they run fully offline in CI.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from data.adapter import CsvReplayAdapter, ReplayConfig
from data.schema import SensorEvent
from data.vibration import generate_vibration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AI4I_HEADER = (
    "UDI,Product ID,Type,Air temperature [K],"
    "Process temperature [K],Rotational speed [rpm],"
    "Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF"
)

AI4I_ROWS = [
    "1,M14860,M,298.1,308.6,1551,42.8,0,0,0,0,0,0,0",
    "2,L47181,L,298.2,308.7,1408,46.3,3,0,0,0,0,0,0",
    "3,L47182,L,298.1,308.5,1498,49.4,5,0,0,0,0,0,0",
    "4,L47183,L,298.2,308.6,1433,39.5,7,0,0,0,0,0,0",
    "5,M14864,M,298.2,308.7,1408,40.0,11,1,0,1,0,0,0",  # HDF failure
]


@pytest.fixture
def mini_csv(tmp_path: Path) -> Path:
    """Write a minimal AI4I-compatible CSV to a temp file."""
    p = tmp_path / "ai4i_mini.csv"
    with open(p, "w", newline="") as f:
        f.write(AI4I_HEADER + "\n")
        for row in AI4I_ROWS:
            f.write(row + "\n")
    return p


# ---------------------------------------------------------------------------
# Vibration generator tests
# ---------------------------------------------------------------------------


class TestVibrationGenerator:
    def test_output_is_float(self):
        v = generate_vibration(
            rotational_speed_rpm=1500, tool_wear_min=50, machine_failure=False, t=0.0
        )
        assert isinstance(v, float)

    def test_no_failure_amplitude_reasonable(self):
        """Without failure, vibration should stay within ±2 m/s²."""
        rng = random.Random(0)
        for t in range(100):
            v = generate_vibration(
                rotational_speed_rpm=1500,
                tool_wear_min=100,
                machine_failure=False,
                t=float(t) * 0.01,
                rng=rng,
            )
            assert abs(v) < 2.0, f"Unexpected amplitude at t={t}: {v}"

    def test_failure_increases_amplitude(self):
        """Failure mode should produce higher mean absolute vibration."""
        rng_ok = random.Random(42)
        rng_fail = random.Random(42)

        normal = [
            abs(
                generate_vibration(
                    rotational_speed_rpm=1500,
                    tool_wear_min=200,
                    machine_failure=False,
                    t=float(i) * 0.01,
                    rng=rng_ok,
                )
            )
            for i in range(200)
        ]
        failed = [
            abs(
                generate_vibration(
                    rotational_speed_rpm=1500,
                    tool_wear_min=200,
                    machine_failure=True,
                    t=float(i) * 0.01,
                    rng=rng_fail,
                )
            )
            for i in range(200)
        ]
        assert sum(failed) / len(failed) > sum(normal) / len(normal)

    def test_reproducible_with_seed(self):
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        v1 = generate_vibration(
            rotational_speed_rpm=1800, tool_wear_min=30, machine_failure=False, t=1.23, rng=rng1
        )
        v2 = generate_vibration(
            rotational_speed_rpm=1800, tool_wear_min=30, machine_failure=False, t=1.23, rng=rng2
        )
        assert v1 == v2


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSensorEventSchema:
    def test_valid_event_parses(self):
        event = SensorEvent(
            event_id="abc-123",
            machine_id="M14860",
            timestamp=datetime.now(UTC),
            air_temperature_k=298.1,
            process_temperature_k=308.6,
            rotational_speed_rpm=1551.0,
            torque_nm=42.8,
            tool_wear_min=0.0,
            vibration_ms2=0.15,
            machine_failure=False,
            failure_type="NONE",
            product_quality="M",
        )
        assert event.machine_id == "M14860"

    def test_process_temp_below_air_raises(self):
        with pytest.raises(Exception):
            SensorEvent(
                event_id="x",
                machine_id="X",
                timestamp=datetime.now(UTC),
                air_temperature_k=300.0,
                process_temperature_k=299.0,  # below air — invalid
                rotational_speed_rpm=1500.0,
                torque_nm=40.0,
                tool_wear_min=0.0,
                vibration_ms2=0.1,
                machine_failure=False,
                failure_type="NONE",
                product_quality="L",
            )

    def test_json_round_trip(self):
        event = SensorEvent(
            event_id="round-trip",
            machine_id="M1",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            air_temperature_k=298.0,
            process_temperature_k=308.0,
            rotational_speed_rpm=1500.0,
            torque_nm=40.0,
            tool_wear_min=0.0,
            vibration_ms2=0.2,
            machine_failure=False,
            failure_type="NONE",
            product_quality="H",
        )
        restored = SensorEvent.model_validate_json(event.model_dump_json())
        assert restored.event_id == event.event_id
        assert restored.timestamp == event.timestamp


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------


class TestCsvReplayAdapter:
    @pytest.mark.asyncio
    async def test_streams_all_rows(self, mini_csv: Path):
        adapter = CsvReplayAdapter(
            csv_path=mini_csv,
            config=ReplayConfig(event_interval_s=0.0, seed=42),
        )
        events = [e async for e in adapter.stream()]
        assert len(events) == len(AI4I_ROWS)

    @pytest.mark.asyncio
    async def test_all_events_are_sensor_events(self, mini_csv: Path):
        adapter = CsvReplayAdapter(
            csv_path=mini_csv,
            config=ReplayConfig(event_interval_s=0.0, seed=42),
        )
        async for event in adapter.stream():
            assert isinstance(event, SensorEvent)

    @pytest.mark.asyncio
    async def test_failure_row_parsed_correctly(self, mini_csv: Path):
        adapter = CsvReplayAdapter(
            csv_path=mini_csv,
            config=ReplayConfig(event_interval_s=0.0, seed=42),
        )
        events = [e async for e in adapter.stream()]
        failure_events = [e for e in events if e.machine_failure]
        assert len(failure_events) == 1
        assert failure_events[0].failure_type == "HDF"

    @pytest.mark.asyncio
    async def test_timestamps_are_monotonically_increasing(self, mini_csv: Path):
        adapter = CsvReplayAdapter(
            csv_path=mini_csv,
            config=ReplayConfig(event_interval_s=0.1, seed=42),
        )
        events = [e async for e in adapter.stream()]
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_missing_file_raises(self, tmp_path: Path):
        adapter = CsvReplayAdapter(
            csv_path=tmp_path / "nonexistent.csv",
            config=ReplayConfig(event_interval_s=0.0),
        )
        with pytest.raises(FileNotFoundError):
            async for _ in adapter.stream():
                pass

    @pytest.mark.asyncio
    async def test_vibration_channel_populated(self, mini_csv: Path):
        adapter = CsvReplayAdapter(
            csv_path=mini_csv,
            config=ReplayConfig(event_interval_s=0.0, seed=1),
        )
        async for event in adapter.stream():
            assert event.vibration_ms2 is not None
            assert isinstance(event.vibration_ms2, float)
