"""
Tests for the Dagster pipeline assets.

Uses Dagster's execute_asset_graph / materialize utilities for unit-level
testing without needing a running Dagster instance.

All tests are fully offline — no MLflow server, no Kafka, no real CSV.
MLflow calls are patched where needed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from dagster import build_asset_context, materialize

from pipelines.assets.features import feature_matrix
from pipelines.assets.anomaly_scores import anomaly_scores, MODEL_FEATURES
from pipelines.assets.alerts import alerts, _severity, _build_alert
from pipelines.resources import AlertSinkResource, MlflowResource


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

AI4I_HEADER = (
    "UDI,Product ID,Type,Air temperature [K],"
    "Process temperature [K],Rotational speed [rpm],"
    "Torque [Nm],Tool wear [min],Machine failure,TWF,HDF,PWF,OSF,RNF"
)
AI4I_ROWS = [
    f"{i},M{10000+i},M,{298+i*0.1:.1f},{308+i*0.1:.1f},{1500+i*10},{40+i},{i*5},0,0,0,0,0,0"
    for i in range(20)
] + ["99,M19999,M,298.5,308.5,1450,75.0,200,1,0,1,0,0,0"]  # one HDF failure


@pytest.fixture
def mini_csv(tmp_path: Path) -> Path:
    p = tmp_path / "ai4i_mini.csv"
    with open(p, "w", newline="") as f:
        f.write(AI4I_HEADER + "\n")
        for row in AI4I_ROWS:
            f.write(row + "\n")
    return p


def _make_raw_df(n: int = 50, n_failures: int = 2) -> pd.DataFrame:
    """Build a synthetic raw_events DataFrame for unit testing downstream assets."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "event_id": [f"evt-{i}" for i in range(n)],
            "machine_id": [f"M{i}" for i in range(n)],
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="s"),
            "air_temperature_k": rng.uniform(295, 305, n),
            "process_temperature_k": rng.uniform(306, 315, n),
            "rotational_speed_rpm": rng.uniform(1168, 2886, n),
            "torque_nm": rng.uniform(3.8, 76.6, n),
            "tool_wear_min": rng.uniform(0, 253, n),
            "vibration_ms2": rng.normal(0.3, 0.05, n),
            "machine_failure": ([True] * n_failures) + ([False] * (n - n_failures)),
            "failure_type": (["HDF"] * n_failures) + (["NONE"] * (n - n_failures)),
            "product_quality": rng.choice(["L", "M", "H"], n).tolist(),
            "source": ["csv_replay"] * n,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------


class TestFeatureMatrix:
    def test_returns_dataframe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        ctx = build_asset_context()
        raw = _make_raw_df()
        result = feature_matrix(ctx, raw)
        assert isinstance(result, pd.DataFrame)

    def test_derived_columns_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        ctx = build_asset_context()
        raw = _make_raw_df()
        result = feature_matrix(ctx, raw)
        for col in ["temp_delta", "power_proxy_kw", "vibration_energy", "wear_ratio"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_rolling_columns_present(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        ctx = build_asset_context()
        raw = _make_raw_df()
        result = feature_matrix(ctx, raw)
        for col in ["vibration_rolling_mean", "vibration_rolling_std", "power_rolling_mean"]:
            assert col in result.columns

    def test_row_count_preserved(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        ctx = build_asset_context()
        raw = _make_raw_df(n=100)
        result = feature_matrix(ctx, raw)
        assert len(result) == 100

    def test_no_nulls_in_features(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        ctx = build_asset_context()
        raw = _make_raw_df(n=50)
        result = feature_matrix(ctx, raw)
        assert result[MODEL_FEATURES].isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Anomaly scoring tests
# ---------------------------------------------------------------------------


class TestAnomalyScores:
    def _make_features(self, n: int = 100, n_failures: int = 4) -> pd.DataFrame:
        raw = _make_raw_df(n=n, n_failures=n_failures)
        # simulate feature engineering without writing files
        import numpy as np
        df = raw.copy()
        df["temp_delta"] = df["process_temperature_k"] - df["air_temperature_k"]
        df["power_proxy_kw"] = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60 / 1000
        df["vibration_energy"] = df["vibration_ms2"] ** 2
        df["wear_ratio"] = df["tool_wear_min"] / 253.0
        df["vibration_rolling_mean"] = df["vibration_ms2"].rolling(10, min_periods=1).mean()
        df["vibration_rolling_std"] = df["vibration_ms2"].rolling(10, min_periods=1).std().fillna(0)
        df["power_rolling_mean"] = df["power_proxy_kw"].rolling(10, min_periods=1).mean()
        return df

    @patch("pipelines.assets.anomaly_scores.mlflow")
    def test_returns_dataframe_with_score_col(self, mock_mlflow, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ctx = build_asset_context(resources={"mlflow_resource": MlflowResource()})
        feat = self._make_features()
        result = anomaly_scores(ctx, feat)
        assert "anomaly_score" in result.columns
        assert "is_anomaly" in result.columns

    @patch("pipelines.assets.anomaly_scores.mlflow")
    def test_scores_in_01_range(self, mock_mlflow, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ctx = build_asset_context(resources={"mlflow_resource": MlflowResource()})
        feat = self._make_features()
        result = anomaly_scores(ctx, feat)
        assert result["anomaly_score"].between(0.0, 1.0).all()

    @patch("pipelines.assets.anomaly_scores.mlflow")
    def test_is_anomaly_is_binary(self, mock_mlflow, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="test-run-id"))
        )
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ctx = build_asset_context(resources={"mlflow_resource": MlflowResource()})
        feat = self._make_features()
        result = anomaly_scores(ctx, feat)
        assert set(result["is_anomaly"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Alert tests
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_severity_bands(self):
        assert _severity(0.95) == "CRITICAL"
        assert _severity(0.80) == "HIGH"
        assert _severity(0.65) == "MEDIUM"
        assert _severity(0.50) == "LOW"

    def test_no_alerts_on_clean_batch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = _make_raw_df(n=50, n_failures=0)
        raw["anomaly_score"] = 0.1
        raw["is_anomaly"] = 0
        ctx = build_asset_context(resources={"alert_sink": AlertSinkResource(sink_type="console")})
        result = alerts(ctx, raw)
        assert result == []

    def test_alerts_returned_for_flagged_events(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = _make_raw_df(n=50, n_failures=3)
        raw["anomaly_score"] = [0.9] * 3 + [0.1] * 47
        raw["is_anomaly"] = [1] * 3 + [0] * 47
        ctx = build_asset_context(resources={"alert_sink": AlertSinkResource(sink_type="console")})
        result = alerts(ctx, raw)
        assert len(result) == 3

    def test_alert_file_written(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        raw = _make_raw_df(n=20, n_failures=2)
        raw["anomaly_score"] = [0.85, 0.90] + [0.1] * 18
        raw["is_anomaly"] = [1, 1] + [0] * 18
        out_path = str(tmp_path / "data" / "processed" / "test_alerts.jsonl")
        ctx = build_asset_context(resources={"alert_sink": AlertSinkResource(sink_type="file", output_path=out_path)})
        alerts(ctx, raw)
        assert Path(out_path).exists()
        lines = Path(out_path).read_text().strip().splitlines()
        assert len(lines) == 2

    def test_alert_has_required_keys(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = _make_raw_df(n=10, n_failures=1)
        raw["anomaly_score"] = [0.88] + [0.1] * 9
        raw["is_anomaly"] = [1] + [0] * 9
        ctx = build_asset_context(resources={"alert_sink": AlertSinkResource(sink_type="console")})
        result = alerts(ctx, raw)
        assert len(result) == 1
        a = result[0]
        for key in ["alert_id", "machine_id", "anomaly_score", "severity", "top_features"]:
            assert key in a
