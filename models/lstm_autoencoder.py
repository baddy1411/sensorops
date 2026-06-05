"""
LSTM Autoencoder anomaly detection model.

Architecture:
  Encoder: LSTM(hidden_size) → latent representation
  Decoder: LSTM(hidden_size) → reconstructed sequence

Anomaly scoring:
  Reconstruction error (MSE per sample) is the anomaly score.
  The model learns the normal operating envelope; deviations from it
  indicate anomalous behaviour.

  Scores are min-max normalised to [0, 1] over the training distribution.

Sequence handling:
  The model expects 2-D input (n_samples, n_features) and wraps each
  sample into a sequence of length `seq_len` (sliding window).

  For inference on a single-step stream, seq_len=1 is valid and degrades
  to a dense autoencoder-like behaviour — still meaningful for our use case.

Dependencies:
  torch (optional — model gracefully raises ImportError if not installed)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from models.base import BaseAnomalyModel


class LSTMAutoencoder(BaseAnomalyModel):
    """
    LSTM Autoencoder for time-series anomaly detection.

    Requires PyTorch. Install with: pip install torch
    """

    model_name = "lstm_autoencoder"

    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 64,
        num_layers: int = 2,
        seq_len: int = 30,
        learning_rate: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 64,
        threshold: float = 0.6,
        device: str = "cpu",
    ) -> None:
        super().__init__(threshold=threshold)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device_str = device

        self._model: Any = None
        self._score_min: float = 0.0
        self._score_max: float = 1.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_torch(self):
        try:
            import torch

            return torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for LSTMAutoencoder.\nInstall with: pip install torch"
            ) from e

    def _build_model(self, torch):
        import torch.nn as nn

        class _Encoder(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0.0,
                )

            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return h[-1]  # last layer hidden state

        class _Decoder(nn.Module):
            def __init__(self, hidden_size, input_size, num_layers, seq_len):
                super().__init__()
                self.seq_len = seq_len
                self.lstm = nn.LSTM(
                    hidden_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, input_size)

            def forward(self, z):
                # Repeat latent vector across time steps
                z_repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)
                out, _ = self.lstm(z_repeated)
                return self.fc(out)

        class _LSTMAEModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, seq_len):
                super().__init__()
                self.encoder = _Encoder(input_size, hidden_size, num_layers)
                self.decoder = _Decoder(hidden_size, input_size, num_layers, seq_len)

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        return _LSTMAEModel(self.input_size, self.hidden_size, self.num_layers, self.seq_len)

    def _make_sequences(self, X: np.ndarray, torch) -> Any:
        """Slide a window of seq_len over X → tensor (n, seq_len, features)."""
        n = len(X)
        if n < self.seq_len:
            # Pad with repetition of first row
            pad = np.tile(X[0], (self.seq_len - n, 1))
            X = np.vstack([pad, X])
            n = self.seq_len

        seqs = []
        for i in range(n - self.seq_len + 1):
            seqs.append(X[i : i + self.seq_len])

        return torch.tensor(np.stack(seqs), dtype=torch.float32)

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Returns per-sample MSE reconstruction error."""
        torch = self._get_torch()
        self._model.eval()
        device = torch.device(self.device_str)
        seqs = self._make_sequences(X, torch).to(device)

        with torch.no_grad():
            recon = self._model(seqs)
            # MSE per sequence
            mse = ((seqs - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()

        # Align back to original n_samples (each sample contributes to seq_len sequences)
        # Simple approach: pad the beginning with the first error value
        n_orig = len(X)
        n_seqs = len(mse)
        pad_size = n_orig - n_seqs
        if pad_size > 0:
            mse = np.concatenate([np.full(pad_size, mse[0]), mse])
        return mse

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> LSTMAutoencoder:
        torch = self._get_torch()
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.device_str)
        self._model = self._build_model(torch).to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        seqs = self._make_sequences(X, torch)
        dataset = TensorDataset(seqs)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = self._model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                avg = epoch_loss / len(loader)
                print(f"[LSTM-AE] epoch {epoch + 1}/{self.n_epochs} loss={avg:.6f}")

        # Calibrate score range on training data
        errors = self._reconstruction_errors(X)
        self._score_min = float(errors.min())
        self._score_max = float(errors.max())
        self._is_fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        errors = self._reconstruction_errors(X)
        lo, hi = self._score_min, self._score_max
        if hi == lo:
            return np.zeros(len(X))
        return np.clip((errors - lo) / (hi - lo), 0.0, 1.0)

    def get_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "seq_len": self.seq_len,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "threshold": self.threshold,
            "device": self.device_str,
        }
