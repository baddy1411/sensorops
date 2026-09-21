"""
Model loader — loads and caches the active anomaly model for serving.

Loading strategy (in priority order):
  1. SENSOROPS_MODEL_PATH env var  → load from local joblib file
  2. MLFLOW_MODEL_URI env var       → load from MLflow registry
     (e.g. models:/sensorops-isolation-forest/Production)
  3. Fallback                       → train a fresh IsolationForest on the AI4I CSV at startup

The loaded model is cached as a module-level singleton so FastAPI workers
share it without re-loading on every request.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import numpy as np

from models.base import BaseAnomalyModel
from models.isolation_forest import IsolationForestModel

_lock = threading.Lock()
_model: BaseAnomalyModel | None = None


def _running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def get_model() -> BaseAnomalyModel:
    """Return the cached model, loading it on first call (thread-safe).

    Safe to call from sync code and from async request handlers once the
    lifespan has warmed the model. If the model isn't loaded yet *and* we're
    inside a running event loop, use ``await init_model()`` instead — blocking
    on ``asyncio.run`` there would raise ``RuntimeError``.
    """
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        if _running_loop():
            raise RuntimeError(
                "model not loaded yet inside a running event loop; "
                "await serving.model_loader.init_model() during startup"
            )
        _model = asyncio.run(_load_model())
    return _model


async def init_model() -> BaseAnomalyModel:
    """Load (or return) the model from async code, e.g. the FastAPI lifespan."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = await _load_model()
    return _model


def reload_model() -> BaseAnomalyModel:
    """Force a reload (useful after a new model is promoted to Production)."""
    global _model
    with _lock:
        if _running_loop():
            raise RuntimeError("await serving.model_loader.areload_model() from async code")
        _model = asyncio.run(_load_model())
    return _model


async def areload_model() -> BaseAnomalyModel:
    """Async variant of :func:`reload_model` for use inside a running loop."""
    global _model
    with _lock:
        _model = await _load_model()
    return _model


async def _load_model() -> BaseAnomalyModel:
    # 1. Local file
    local_path = os.getenv("SENSOROPS_MODEL_PATH", "")
    if local_path and Path(local_path).exists():
        print(f"[loader] loading model from file: {local_path}")
        return IsolationForestModel.load(local_path)

    # 2. MLflow URI
    mlflow_uri = os.getenv("MLFLOW_MODEL_URI", "")
    if mlflow_uri:
        print(f"[loader] loading model from MLflow: {mlflow_uri}")
        try:
            import mlflow.sklearn

            return mlflow.sklearn.load_model(mlflow_uri)
        except Exception as exc:
            print(f"[loader] MLflow load failed ({exc}), falling back to default")

    # 3. Fallback: train fresh IF on available data
    print("[loader] no model found — training default IsolationForest")
    return await _train_fallback_model()


async def _train_fallback_model() -> IsolationForestModel:
    """Train a baseline IF on whatever data is available locally."""
    import pandas as pd

    csv_path = os.getenv("SENSOROPS_CSV_PATH", "data/raw/ai4i2020.csv")

    if Path(csv_path).exists():
        # NB: dagster is optional — the lean serving image uses the no-op
        # shims from pipelines.dagster_compat so the asset *functions* stay
        # callable without the orchestrator installed.
        from data.adapter import CsvReplayAdapter, ReplayConfig
        from pipelines.assets.anomaly_scores import MODEL_FEATURES
        from pipelines.assets.features import feature_matrix as _feat_fn
        from pipelines.dagster_compat import build_asset_context

        adapter = CsvReplayAdapter(csv_path=csv_path, config=ReplayConfig(event_interval_s=0.0))

        async def _collect():
            events = []
            async for e in adapter.stream():
                events.append(e.model_dump())
                if len(events) >= 2000:
                    break
            return events

        rows = await _collect()
        raw_df = pd.DataFrame(rows)
        feat_df = _feat_fn(build_asset_context(), raw_df)
        X = feat_df[MODEL_FEATURES].values
    else:
        print("[loader] no CSV found — using random normal data for fallback model")
        X = np.random.default_rng(42).normal(size=(500, 13))

    model = IsolationForestModel(n_estimators=100, contamination=0.035)
    model.fit(X)
    return model
