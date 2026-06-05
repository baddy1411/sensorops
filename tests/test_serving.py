"""
Tests for the FastAPI serving layer.

Uses FastAPI's TestClient — no running server needed.
The model loader is patched to inject a pre-fitted IsolationForest
so tests are fast and fully offline.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from models.isolation_forest import IsolationForestModel
from serving.schemas import _score_to_severity


# ---------------------------------------------------------------------------
# Shared fixture — pre-fitted model injected into the loader
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_model() -> IsolationForestModel:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 13))
    model = IsolationForestModel(contamination=0.035)
    model.fit(X)
    return model


@pytest.fixture(scope="module")
def client(fitted_model):
    with patch("serving.model_loader.get_model", return_value=fitted_model), \
         patch("serving.model_loader._model", fitted_model):
        from serving.app import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_health_has_version(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.get("/health")
        assert "version" in resp.json()


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


VALID_READING = {
    "reading": {
        "machine_id": "M14860",
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551.0,
        "torque_nm": 42.8,
        "tool_wear_min": 0.0,
        "vibration_ms2": 0.15,
    }
}


class TestPredict:
    def test_predict_returns_200(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict", json=VALID_READING)
        assert resp.status_code == 200

    def test_predict_response_shape(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict", json=VALID_READING)
        body = resp.json()
        assert "result" in body
        assert "model_name" in body
        r = body["result"]
        assert "anomaly_score" in r
        assert "is_anomaly" in r
        assert "severity" in r

    def test_predict_score_in_range(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict", json=VALID_READING)
        score = resp.json()["result"]["anomaly_score"]
        assert 0.0 <= score <= 1.0

    def test_predict_machine_id_echoed(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict", json=VALID_READING)
        assert resp.json()["result"]["machine_id"] == "M14860"

    def test_predict_invalid_temp_returns_422(self, client, fitted_model):
        bad = {
            "reading": {**VALID_READING["reading"], "air_temperature_k": 999.9}
        }
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


class TestBatchPredict:
    def _batch_body(self, n: int) -> dict:
        return {
            "readings": [VALID_READING["reading"]] * n
        }

    def test_batch_single_item(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict/batch", json=self._batch_body(1))
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1

    def test_batch_multiple_items(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict/batch", json=self._batch_body(10))
        body = resp.json()
        assert body["total"] == 10
        assert len(body["results"]) == 10

    def test_batch_anomaly_count_consistent(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict/batch", json=self._batch_body(20))
        body = resp.json()
        actual_count = sum(1 for r in body["results"] if r["is_anomaly"])
        assert actual_count == body["anomaly_count"]

    def test_batch_empty_returns_422(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.post("/api/v1/predict/batch", json={"readings": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_model_info_returns_200(self, client, fitted_model):
        with patch("serving.app.get_model", return_value=fitted_model):
            resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        body = resp.json()
        assert "model_name" in body
        assert "params" in body
        assert body["is_fitted"] is True


# ---------------------------------------------------------------------------
# Schema unit tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_severity_critical(self):
        assert _score_to_severity(0.95) == "CRITICAL"

    def test_severity_high(self):
        assert _score_to_severity(0.80) == "HIGH"

    def test_severity_medium(self):
        assert _score_to_severity(0.65) == "MEDIUM"

    def test_severity_low(self):
        assert _score_to_severity(0.40) == "LOW"
