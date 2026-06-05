"""
MLflow Model Registry helpers for SensorOps.

Responsibilities:
  - Log any BaseAnomalyModel to an MLflow run with full params + metrics
  - Promote a model version to Staging or Production based on F1 threshold
  - Compare multiple models and return the winner by roc_auc
  - Fetch the current Production model for serving

Promotion policy (Industrie 4.0 context):
  A model must achieve F1 >= 0.50 and ROC-AUC >= 0.75 to be promoted
  to Staging. Production promotion requires manual approval (EU AI Act
  compliance — automated promotion to Production is disabled).
"""

from __future__ import annotations

from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np

from models.base import BaseAnomalyModel

# Thresholds for automatic Staging promotion
STAGING_MIN_F1 = 0.50
STAGING_MIN_ROC_AUC = 0.75


def log_model_run(
    model: BaseAnomalyModel,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> tuple[str, dict[str, float]]:
    """
    Train model on X_train, evaluate on (X_test, y_test), log everything to MLflow.

    Returns (run_id, metrics_dict).
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = run_name or f"{model.model_name}-run"
    tags = {"model_type": model.model_name, **(extra_tags or {})}

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        # Params
        mlflow.log_params(model.get_params())
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Train
        model.fit(X_train)

        # Metrics
        metrics = model.evaluate(X_test, y_test)
        mlflow.log_metrics(metrics)

        # Flag rate on test set
        flag_rate = float(model.predict(X_test).mean())
        mlflow.log_metric("flag_rate", flag_rate)

        # Log model artifact (sklearn-compatible via joblib pyfunc)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model.model_name,
            registered_model_name=registered_model_name,
        )

        run_id = run.info.run_id

    return run_id, metrics


def promote_to_staging(
    registered_model_name: str,
    run_id: str,
    metrics: dict[str, float],
    tracking_uri: str,
    dry_run: bool = False,
) -> bool:
    """
    Promote the model version from run_id to Staging if it meets quality gates.

    Returns True if promoted, False if gates not met.
    dry_run=True logs the decision without actually transitioning.
    """
    f1 = metrics.get("f1", 0.0)
    roc_auc = metrics.get("roc_auc", 0.0)

    gates_passed = f1 >= STAGING_MIN_F1 and roc_auc >= STAGING_MIN_ROC_AUC

    print(
        f"[registry] promotion check — "
        f"F1={f1:.3f} (min {STAGING_MIN_F1}), "
        f"ROC-AUC={roc_auc:.3f} (min {STAGING_MIN_ROC_AUC}) — "
        f"{'PASS' if gates_passed else 'FAIL'}"
    )

    if not gates_passed or dry_run:
        return gates_passed

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    # Find the version registered from this run
    versions = client.search_model_versions(f"name='{registered_model_name}'")
    target = next(
        (v for v in versions if v.run_id == run_id),
        None,
    )
    if target is None:
        print(f"[registry] no version found for run_id={run_id}")
        return False

    client.transition_model_version_stage(
        name=registered_model_name,
        version=target.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    print(f"[registry] version {target.version} of '{registered_model_name}' promoted to Staging")
    return True


def compare_models(
    models: list[BaseAnomalyModel],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[BaseAnomalyModel, dict[str, dict[str, float]]]:
    """
    Fit and evaluate all models, return (best_model, all_metrics).

    Best model is chosen by ROC-AUC (ties broken by F1).
    """
    results: dict[str, dict[str, float]] = {}

    for model in models:
        model.fit(X_train)
        metrics = model.evaluate(X_test, y_test)
        # Use id() to disambiguate duplicate model names in the same comparison run
        key = f"{model.model_name}_{id(model)}"
        results[key] = metrics
        print(
            f"[compare] {model.model_name:25s} "
            f"F1={metrics['f1']:.3f}  "
            f"ROC-AUC={metrics.get('roc_auc', float('nan')):.3f}  "
            f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}"
        )

    best = max(
        models,
        key=lambda m: (
            results[f"{m.model_name}_{id(m)}"].get("roc_auc", 0.0),
            results[f"{m.model_name}_{id(m)}"]["f1"],
        ),
    )
    print(f"[compare] winner → {best.model_name}")
    return best, results


def load_production_model(
    registered_model_name: str,
    tracking_uri: str,
) -> Any:
    """
    Load the current Production model from MLflow registry.

    Returns the raw Python model object (sklearn-compatible).
    Raises ValueError if no Production version exists.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    versions = client.get_latest_versions(registered_model_name, stages=["Production"])
    if not versions:
        raise ValueError(
            f"No Production version found for model '{registered_model_name}'. "
            "Promote a Staging model to Production first (requires human approval)."
        )

    v = versions[0]
    model_uri = f"models:/{registered_model_name}/Production"
    print(f"[registry] loading Production version {v.version} from {model_uri}")
    return mlflow.sklearn.load_model(model_uri)
