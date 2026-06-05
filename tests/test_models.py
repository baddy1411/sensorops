"""
Tests for all three anomaly detection model wrappers.

Tests run fully offline:
  - IsolationForestModel  — no extra deps
  - LSTMAutoencoder       — skipped if torch not installed
  - EchoStateNetworkModel — skipped if reservoirpy not installed
  - registry.compare_models — uses IF only for speed
"""

from __future__ import annotations

import numpy as np
import pytest

from models.base import BaseAnomalyModel
from models.isolation_forest import IsolationForestModel
from models.registry import compare_models

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _normal_data(n: int = 300, features: int = 13, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n, features))


def _anomaly_data(n: int = 30, features: int = 13, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=5.0, scale=1.0, size=(n, features))


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class TestBaseInterface:
    def test_isolation_forest_is_base_anomaly_model(self):
        assert isinstance(IsolationForestModel(), BaseAnomalyModel)

    def test_predict_raises_before_fit(self):
        model = IsolationForestModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.score(_normal_data(10))


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------


class TestIsolationForest:
    def test_fit_returns_self(self):
        model = IsolationForestModel()
        X = _normal_data()
        result = model.fit(X)
        assert result is model

    def test_scores_in_01(self):
        model = IsolationForestModel()
        X = _normal_data()
        model.fit(X)
        scores = model.score(X)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_anomalies_score_higher_than_normal(self):
        model = IsolationForestModel(contamination=0.1)
        X_normal = _normal_data(n=400)
        X_anomaly = _anomaly_data(n=40)
        model.fit(X_normal)
        normal_mean = model.score(X_normal).mean()
        anomaly_mean = model.score(X_anomaly).mean()
        assert anomaly_mean > normal_mean, (
            f"Anomaly mean score {anomaly_mean:.3f} should exceed normal mean {normal_mean:.3f}"
        )

    def test_predict_returns_binary(self):
        model = IsolationForestModel()
        X = _normal_data()
        model.fit(X)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_evaluate_returns_expected_keys(self):
        model = IsolationForestModel(contamination=0.1)
        X_train = _normal_data(n=300)
        X_test = np.vstack([_normal_data(n=50), _anomaly_data(n=20)])
        y_test = np.array([0] * 50 + [1] * 20)
        model.fit(X_train)
        metrics = model.evaluate(X_test, y_test)
        for key in ["precision", "recall", "f1", "roc_auc"]:
            assert key in metrics

    def test_get_params_has_model_name(self):
        model = IsolationForestModel()
        params = model.get_params()
        assert params["model_name"] == "isolation_forest"

    def test_save_and_load(self, tmp_path):
        model = IsolationForestModel()
        X = _normal_data()
        model.fit(X)
        path = tmp_path / "if_model.joblib"
        model.save(path)
        loaded = IsolationForestModel.load(path)
        scores_orig = model.score(X)
        scores_loaded = loaded.score(X)
        np.testing.assert_array_almost_equal(scores_orig, scores_loaded)


# ---------------------------------------------------------------------------
# LSTM Autoencoder (optional — skip if torch not installed)
# ---------------------------------------------------------------------------


torch_available = pytest.importorskip  # used per test


class TestLSTMAutoencoder:
    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        pytest.importorskip("torch", reason="PyTorch not installed")

    def test_fit_and_score(self):
        from models.lstm_autoencoder import LSTMAutoencoder

        model = LSTMAutoencoder(
            input_size=13, hidden_size=16, num_layers=1, seq_len=5, n_epochs=2, batch_size=32
        )
        X = _normal_data(n=100)
        model.fit(X)
        scores = model.score(X)
        assert scores.shape == (100,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_anomalies_score_higher(self):
        from models.lstm_autoencoder import LSTMAutoencoder

        model = LSTMAutoencoder(
            input_size=13, hidden_size=16, num_layers=1, seq_len=5, n_epochs=3, batch_size=32
        )
        X_normal = _normal_data(n=200)
        X_anomaly = _anomaly_data(n=50)
        model.fit(X_normal)
        assert model.score(X_anomaly).mean() > model.score(X_normal).mean()

    def test_get_params(self):
        from models.lstm_autoencoder import LSTMAutoencoder

        params = LSTMAutoencoder().get_params()
        assert params["model_name"] == "lstm_autoencoder"


# ---------------------------------------------------------------------------
# Echo State Network (optional — skip if reservoirpy not installed)
# ---------------------------------------------------------------------------


class TestESN:
    @pytest.fixture(autouse=True)
    def skip_if_no_reservoirpy(self):
        pytest.importorskip("reservoirpy", reason="reservoirpy not installed")

    def test_fit_and_score(self):
        from models.esn import EchoStateNetworkModel

        model = EchoStateNetworkModel(units=50, spectral_radius=0.9)
        X = _normal_data(n=200)
        model.fit(X)
        scores = model.score(X)
        assert scores.shape == (200,)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_anomalies_score_higher(self):
        from models.esn import EchoStateNetworkModel

        model = EchoStateNetworkModel(units=50)
        X_normal = _normal_data(n=200)
        X_anomaly = _anomaly_data(n=50)
        model.fit(X_normal)
        assert model.score(X_anomaly).mean() > model.score(X_normal).mean()

    def test_get_params_has_thesis_reference(self):
        from models.esn import EchoStateNetworkModel

        params = EchoStateNetworkModel().get_params()
        assert "thesis_reference" in params
        assert "NRMSE" in params["thesis_reference"]


# ---------------------------------------------------------------------------
# Registry / compare_models (no MLflow server needed — just the logic)
# ---------------------------------------------------------------------------


class TestCompareModels:
    def test_returns_best_model_and_metrics(self):
        X_train = _normal_data(n=300)
        X_test = np.vstack([_normal_data(n=50), _anomaly_data(n=20)])
        y_test = np.array([0] * 50 + [1] * 20)

        models = [
            IsolationForestModel(contamination=0.1),
            IsolationForestModel(contamination=0.2),  # second variant
        ]
        best, all_metrics = compare_models(models, X_train, X_test, y_test)
        assert isinstance(best, BaseAnomalyModel)
        assert len(all_metrics) == 2
        for name, m in all_metrics.items():
            assert "f1" in m
            assert "roc_auc" in m
