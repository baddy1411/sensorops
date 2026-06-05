"""
Base class for all SensorOps anomaly detection models.

Every model must implement:
  fit(X)          — train on normal operating data (unsupervised)
  score(X)        — return anomaly scores [0, 1], higher = more anomalous
  predict(X)      — return binary labels {0, 1} using self.threshold
  get_params()    — return dict of hyperparameters for MLflow logging
  model_name      — class-level string identifier

Optional:
  save(path) / load(path) — serialisation (defaults to pickle via joblib)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseAnomalyModel(ABC):
    """Abstract base for unsupervised anomaly detection models."""

    model_name: str = "base"

    def __init__(self, threshold: float = 0.6) -> None:
        """
        Parameters
        ----------
        threshold : Score cutoff above which an event is flagged as anomalous.
        """
        self.threshold = threshold
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> BaseAnomalyModel:
        """Train the model. X shape: (n_samples, n_features)."""
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1]. Shape: (n_samples,)."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly labels using self.threshold."""
        return (self.score(X) >= self.threshold).astype(int)

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return hyperparameter dict for MLflow logging."""
        ...

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseAnomalyModel:
        return joblib.load(path)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.model_name} must be fitted before calling score().")

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """
        Compute evaluation metrics against ground truth labels.

        Returns precision, recall, f1, and roc_auc.
        Assumes binary y_true {0, 1}.
        """
        from sklearn.metrics import (
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        scores = self.score(X)
        y_pred = (scores >= self.threshold).astype(int)

        metrics: dict[str, float] = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        # roc_auc needs both classes present
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        else:
            metrics["roc_auc"] = float("nan")

        return metrics
