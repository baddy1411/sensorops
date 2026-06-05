"""
Asset: model_comparison

Trains and evaluates all three anomaly detection models (Isolation Forest,
LSTM Autoencoder, ESN) against the feature matrix, logs every run to MLflow,
and promotes the best model to MLflow Staging if it passes quality gates.

This asset is intentionally separate from the main scoring pipeline —
it is a heavier, less frequent operation (run on demand or weekly) rather
than every pipeline execution.

Dagster asset key : model_comparison
Group             : model_selection
Input             : feature_matrix (DataFrame)
Output            : dict with model names → metrics + winner
"""

from __future__ import annotations

import pandas as pd
from dagster import asset

from models.esn import EchoStateNetworkModel
from models.isolation_forest import IsolationForestModel
from models.lstm_autoencoder import LSTMAutoencoder
from models.registry import log_model_run, promote_to_staging
from pipelines.assets.anomaly_scores import MODEL_FEATURES
from pipelines.resources import MlflowResource

# Train/test split ratio (time-ordered — no random shuffle for time-series)
TRAIN_RATIO = 0.8


# Model list to compare
def _get_models() -> list:
    return [
        IsolationForestModel(n_estimators=200, contamination=0.035),
        LSTMAutoencoder(input_size=len(MODEL_FEATURES), hidden_size=64, n_epochs=10),
        EchoStateNetworkModel(units=300, spectral_radius=0.9, leak_rate=0.3),
    ]


@asset(
    group_name="model_selection",
    description=(
        "Trains IF, LSTM-AE, and ESN; compares by ROC-AUC; "
        "promotes winner to MLflow Staging if quality gates pass."
    ),
)
def model_comparison(
    context,
    feature_matrix: pd.DataFrame,
    mlflow_resource: MlflowResource,
) -> dict:
    """
    Full model comparison pipeline:
      1. Time-ordered train/test split
      2. Fit and score all three models
      3. Log each to MLflow
      4. Promote best to Staging if F1 >= 0.50 and ROC-AUC >= 0.75
    """
    X_all = feature_matrix[MODEL_FEATURES].values
    y_all = feature_matrix["machine_failure"].astype(int).values

    split = int(len(X_all) * TRAIN_RATIO)
    X_train, X_test = X_all[:split], X_all[split:]
    y_test = y_all[split:]

    # Train on clean-ish data — ideally only normal samples.
    # In practice we use all training data; IF/ESN are robust to low contamination.
    context.log.info(
        f"model_comparison: train={len(X_train)}, test={len(X_test)}, test failures={y_test.sum()}"
    )

    all_metrics: dict[str, dict] = {}
    run_ids: dict[str, str] = {}

    for model in _get_models():
        context.log.info(f"Training {model.model_name}...")
        try:
            run_id, metrics = log_model_run(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                tracking_uri=mlflow_resource.tracking_uri,
                experiment_name=mlflow_resource.experiment_name,
                registered_model_name=f"sensorops-{model.model_name}",
                run_name=f"{model.model_name}-comparison",
                extra_tags={"pipeline_step": "model_comparison"},
            )
            all_metrics[model.model_name] = metrics
            run_ids[model.model_name] = run_id
            context.log.info(
                f"{model.model_name}: F1={metrics['f1']:.3f} "
                f"ROC-AUC={metrics.get('roc_auc', float('nan')):.3f}"
            )
        except Exception as exc:
            context.log.warning(f"{model.model_name} failed: {exc}")
            all_metrics[model.model_name] = {"error": str(exc)}

    # Determine winner by ROC-AUC
    scored = {
        name: m
        for name, m in all_metrics.items()
        if "roc_auc" in m and not isinstance(m.get("roc_auc"), str)
    }

    winner_name = max(scored, key=lambda n: (scored[n].get("roc_auc", 0), scored[n].get("f1", 0)))
    winner_metrics = scored[winner_name]
    winner_run_id = run_ids.get(winner_name, "")

    context.log.info(f"Winner: {winner_name} (ROC-AUC={winner_metrics.get('roc_auc', 0):.3f})")

    # Attempt Staging promotion
    promoted = promote_to_staging(
        registered_model_name=f"sensorops-{winner_name}",
        run_id=winner_run_id,
        metrics=winner_metrics,
        tracking_uri=mlflow_resource.tracking_uri,
    )

    result = {
        "winner": winner_name,
        "winner_metrics": winner_metrics,
        "all_metrics": all_metrics,
        "promoted_to_staging": promoted,
    }

    context.add_output_metadata(
        {
            "winner": winner_name,
            "winner_f1": round(winner_metrics.get("f1", 0), 4),
            "winner_roc_auc": round(winner_metrics.get("roc_auc", 0), 4),
            "promoted_to_staging": promoted,
            "models_evaluated": list(all_metrics.keys()),
        }
    )

    return result
