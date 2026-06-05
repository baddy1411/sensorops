"""
Echo State Network (ESN) anomaly detection model.

Research context:
  This model is directly motivated by the author's M.Sc. thesis on
  Quantum Reservoir Computing vs classical Echo State Networks for
  time-series forecasting (Hénon map benchmark).

  Thesis result: tuned ESN achieves NRMSE 0.0111 vs QRC's 0.0130
  at matched computational resources — ESN wins on both accuracy and
  efficiency. This makes ESN a well-justified choice here, not a toy.

Architecture:
  reservoirpy ESN:
    - Reservoir of N recurrent units with sparse random connectivity
    - Input weights W_in, recurrent weights W (spectral radius < 1 for ESP)
    - Linear readout trained by Ridge regression (closed-form, fast)

Anomaly scoring:
  1. Train on normal data: ESN learns to predict x(t+1) from x(t).
  2. At inference: anomaly score = normalised 1-step prediction error.
     High prediction error → input is out-of-distribution → anomaly.

  This is the reconstruction/forecasting error paradigm used in
  reservoir computing literature for time-series anomaly detection.

Dependencies:
  reservoirpy>=0.3  (pip install reservoirpy)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseAnomalyModel


class EchoStateNetworkModel(BaseAnomalyModel):
    """
    ESN-based anomaly detector using 1-step-ahead prediction error.

    Requires reservoirpy. Install with: pip install reservoirpy
    """

    model_name = "echo_state_network"

    def __init__(
        self,
        units: int = 500,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        leak_rate: float = 0.3,
        connectivity: float = 0.1,
        ridge: float = 1e-6,
        warmup: int = 50,
        threshold: float = 0.6,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        units           : Reservoir size (number of recurrent units).
        spectral_radius : Controls memory capacity. < 1 ensures Echo State Property.
                          Thesis found ρ≈0.9 optimal for Hénon map; reused here.
        input_scaling   : Scales input weight matrix W_in.
        leak_rate       : Leaky-integration rate α ∈ (0, 1].
                          Lower → longer memory. 0.3 is a good starting point.
        connectivity    : Reservoir weight matrix sparsity.
        ridge           : L2 regularisation for the linear readout.
        warmup          : Discarded initial timesteps (transient washout).
        seed            : Random seed for reservoir weight initialisation.
        """
        super().__init__(threshold=threshold)
        self.units = units
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.connectivity = connectivity
        self.ridge = ridge
        self.warmup = warmup
        self.seed = seed

        self._reservoir: Any = None
        self._readout: Any = None
        self._esn: Any = None
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    def _build_esn(self, rpy):
        """Construct reservoirpy ESN: Reservoir → Ridge readout."""
        reservoir = rpy.nodes.Reservoir(
            units=self.units,
            sr=self.spectral_radius,
            input_scaling=self.input_scaling,
            lr=self.leak_rate,
            rc_connectivity=self.connectivity,
            seed=self.seed,
        )
        readout = rpy.nodes.Ridge(ridge=self.ridge)
        return reservoir >> readout

    def _get_rpy(self):
        try:
            import reservoirpy as rpy
            rpy.set_seed(self.seed)
            # verbosity() removed in reservoirpy v0.4 — use set_verbosity if available
            if hasattr(rpy, "set_verbosity"):
                rpy.set_verbosity(0)
            return rpy
        except ImportError as e:
            raise ImportError(
                "reservoirpy is required for EchoStateNetworkModel.\n"
                "Install with: pip install reservoirpy"
            ) from e

    def fit(self, X: np.ndarray) -> "EchoStateNetworkModel":
        """
        Train ESN as a 1-step-ahead forecaster on normal data.

        X shape: (n_samples, n_features)
        Training target: x(t+1) given x(t), i.e., X[1:] given X[:-1].
        """
        rpy = self._get_rpy()
        self._esn = self._build_esn(rpy)

        # Shift: input = X[:-1], target = X[1:]
        X_in = X[:-1]
        Y_target = X[1:]

        self._esn.fit(X_in, Y_target, warmup=self.warmup)

        # Calibrate error range on training data
        errors = self._raw_errors(X)
        self._score_min = float(errors.min())
        self._score_max = float(errors.max())
        self._is_fitted = True
        return self

    def _raw_errors(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MAE prediction error."""
        if len(X) < 2:
            return np.zeros(len(X))

        X_in = X[:-1]
        Y_hat = self._esn.run(X_in)          # shape: (n-1, n_features)
        Y_true = X[1:]
        errors = np.abs(Y_hat - Y_true).mean(axis=1)  # MAE per step

        # Pad first sample with mean error (no prediction available for t=0)
        return np.concatenate([[errors.mean()], errors])

    def score(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        errors = self._raw_errors(X)
        lo, hi = self._score_min, self._score_max
        if hi == lo:
            return np.zeros(len(X))
        return np.clip((errors - lo) / (hi - lo), 0.0, 1.0)

    def get_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "units": self.units,
            "spectral_radius": self.spectral_radius,
            "input_scaling": self.input_scaling,
            "leak_rate": self.leak_rate,
            "connectivity": self.connectivity,
            "ridge": self.ridge,
            "warmup": self.warmup,
            "threshold": self.threshold,
            "seed": self.seed,
            "thesis_reference": "NRMSE=0.0111 (ESN) vs 0.0130 (QRC) on Hénon map",
        }
