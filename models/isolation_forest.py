"""
Isolation Forest anomaly detection model wrapper.

Wraps sklearn's IsolationForest to conform to BaseAnomalyModel.

Why Isolation Forest:
  - No labels required (fully unsupervised)
  - O(n log n) training, fast inference
  - Handles high-dimensional tabular sensor data well
  - Contamination param can be set to known failure rate (~3.5% in AI4I)

Score conversion:
  sklearn's decision_function returns negative values for anomalies.
  We negate and min-max scale to [0, 1] so higher = more anomalous,
  consistent with all other SensorOps models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from models.base import BaseAnomalyModel


class IsolationForestModel(BaseAnomalyModel):
    """Isolation Forest wrapper — baseline anomaly detector."""

    model_name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.035,
        max_samples: str | int = "auto",
        random_state: int = 42,
        threshold: float = 0.6,
    ) -> None:
        super().__init__(threshold=threshold)
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        self._model.fit(X)
        # Calibrate scaling range on training data
        raw = -self._model.decision_function(X)
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        self._is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        raw = -self._model.decision_function(X)
        lo, hi = self._score_min, self._score_max
        if hi == lo:
            return np.zeros(len(X))
        return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)

    def get_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": str(self.max_samples),
            "random_state": self.random_state,
            "threshold": self.threshold,
        }
