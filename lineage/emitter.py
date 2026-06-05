"""
OpenLineage emitter for SensorOps.

Tracks the full data lineage graph:
  raw_events (CSV) → feature_matrix → anomaly_scores → alerts → incident_reports

Each pipeline run emits START and COMPLETE events to the OpenLineage backend
(Marquez by default, or any OL-compatible server).

In local/offline mode (OPENLINEAGE_URL not set) events are written to a
JSONL file — useful for development and EU AI Act audit export.

OpenLineage concepts used:
  Run     — one execution of a pipeline step
  Job     — the transformation (e.g. "feature_engineering")
  Dataset — input or output data (e.g. "raw_events.parquet")
  Facets  — typed metadata attached to runs/datasets/jobs
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Lightweight OL event builder
# (Uses openlineage-python if available, falls back to raw dict construction)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_run_id() -> str:
    return str(uuid.uuid4())


def _dataset(namespace: str, name: str, facets: dict | None = None) -> dict:
    d: dict[str, Any] = {
        "_producer": "https://github.com/yourusername/sensorops",
        "_schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/$defs/InputDataset",
        "namespace": namespace,
        "name": name,
    }
    if facets:
        d["facets"] = facets
    return d


def _schema_facet(fields: list[dict]) -> dict:
    return {
        "schema": {
            "_producer": "sensorops",
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
            "fields": fields,
        }
    }


def _data_quality_facet(row_count: int, null_count: int = 0) -> dict:
    return {
        "dataQualityMetrics": {
            "_producer": "sensorops",
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DataQualityMetricsDatasetFacet.json",
            "rowCount": row_count,
            "nullCount": null_count,
        }
    }


def _file_hash_facet(path: str) -> dict:
    try:
        data = Path(path).read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
    except FileNotFoundError:
        sha256 = "unavailable"
    return {
        "fileHash": {
            "_producer": "sensorops",
            "algorithm": "sha256",
            "hash": sha256,
            "path": path,
        }
    }


# ---------------------------------------------------------------------------
# LineageEmitter
# ---------------------------------------------------------------------------


class LineageEmitter:
    """
    Emits OpenLineage events for each SensorOps pipeline step.

    Usage:
        emitter = LineageEmitter()
        run_id = emitter.emit_start("feature_engineering", inputs=[...])
        # ... do work ...
        emitter.emit_complete(run_id, "feature_engineering", inputs=[...], outputs=[...])
    """

    NAMESPACE = "sensorops"

    # Pre-defined dataset schemas matching the pipeline
    SCHEMAS = {
        "raw_events": [
            {"name": "event_id", "type": "STRING"},
            {"name": "machine_id", "type": "STRING"},
            {"name": "timestamp", "type": "TIMESTAMP"},
            {"name": "air_temperature_k", "type": "DOUBLE"},
            {"name": "process_temperature_k", "type": "DOUBLE"},
            {"name": "rotational_speed_rpm", "type": "DOUBLE"},
            {"name": "torque_nm", "type": "DOUBLE"},
            {"name": "tool_wear_min", "type": "DOUBLE"},
            {"name": "vibration_ms2", "type": "DOUBLE"},
            {"name": "machine_failure", "type": "BOOLEAN"},
            {"name": "failure_type", "type": "STRING"},
        ],
        "features": [
            {"name": "temp_delta", "type": "DOUBLE"},
            {"name": "power_proxy_kw", "type": "DOUBLE"},
            {"name": "vibration_energy", "type": "DOUBLE"},
            {"name": "wear_ratio", "type": "DOUBLE"},
            {"name": "vibration_rolling_mean", "type": "DOUBLE"},
            {"name": "vibration_rolling_std", "type": "DOUBLE"},
        ],
        "anomaly_scores": [
            {"name": "anomaly_score", "type": "DOUBLE"},
            {"name": "is_anomaly", "type": "INTEGER"},
        ],
        "alerts": [
            {"name": "alert_id", "type": "STRING"},
            {"name": "severity", "type": "STRING"},
            {"name": "probable_cause", "type": "STRING"},
        ],
    }

    def __init__(
        self,
        url: str | None = None,
        fallback_log: str = "data/processed/lineage.jsonl",
    ) -> None:
        self._url = url or os.getenv("OPENLINEAGE_URL", "")
        self._fallback_log = Path(fallback_log)
        self._ol_client = self._build_client()

    def _build_client(self):
        """Try to build an openlineage-python client; return None on failure."""
        if not self._url:
            return None
        try:
            from openlineage.client import OpenLineageClient
            return OpenLineageClient(url=self._url)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def emit_start(
        self,
        job_name: str,
        inputs: list[str] | None = None,
        run_id: str | None = None,
        extra_facets: dict | None = None,
    ) -> str:
        """
        Emit a RUN_START event. Returns the run_id for pairing with emit_complete.
        """
        run_id = run_id or _make_run_id()
        event = self._build_event(
            event_type="START",
            job_name=job_name,
            run_id=run_id,
            inputs=inputs or [],
            outputs=[],
            extra_facets=extra_facets,
        )
        self._send(event)
        return run_id

    def emit_complete(
        self,
        run_id: str,
        job_name: str,
        inputs: list[str],
        outputs: list[str],
        row_counts: dict[str, int] | None = None,
        extra_facets: dict | None = None,
    ) -> None:
        """Emit a RUN_COMPLETE event."""
        event = self._build_event(
            event_type="COMPLETE",
            job_name=job_name,
            run_id=run_id,
            inputs=inputs,
            outputs=outputs,
            row_counts=row_counts or {},
            extra_facets=extra_facets,
        )
        self._send(event)

    def emit_fail(self, run_id: str, job_name: str, error: str) -> None:
        """Emit a RUN_FAIL event."""
        event = self._build_event(
            event_type="FAIL",
            job_name=job_name,
            run_id=run_id,
            inputs=[],
            outputs=[],
            extra_facets={"errorMessage": {"message": error}},
        )
        self._send(event)

    # ------------------------------------------------------------------
    # Convenience methods for each pipeline step
    # ------------------------------------------------------------------

    def track_ingest(self, run_id: str, csv_path: str, row_count: int) -> None:
        facets = _file_hash_facet(csv_path)
        self.emit_complete(
            run_id=run_id,
            job_name="ingest.raw_events",
            inputs=[csv_path],
            outputs=["data/processed/raw_events.parquet"],
            row_counts={"raw_events": row_count},
            extra_facets=facets,
        )

    def track_feature_engineering(
        self, run_id: str, input_rows: int, output_rows: int
    ) -> None:
        self.emit_complete(
            run_id=run_id,
            job_name="features.feature_matrix",
            inputs=["data/processed/raw_events.parquet"],
            outputs=["data/processed/features.parquet"],
            row_counts={"features": output_rows},
        )

    def track_scoring(
        self,
        run_id: str,
        model_name: str,
        model_version: str,
        dataset_hash: str,
        n_samples: int,
        n_flagged: int,
        mlflow_run_id: str,
    ) -> None:
        self.emit_complete(
            run_id=run_id,
            job_name="scoring.anomaly_scores",
            inputs=["data/processed/features.parquet"],
            outputs=["data/processed/anomaly_scores.parquet"],
            row_counts={"anomaly_scores": n_samples},
            extra_facets={
                # EU AI Act compliance facet
                "euAiActAudit": {
                    "modelName": model_name,
                    "modelVersion": model_version,
                    "datasetHash": dataset_hash,
                    "nSamples": n_samples,
                    "nFlagged": n_flagged,
                    "flagRate": round(n_flagged / n_samples, 4) if n_samples else 0,
                    "mlflowRunId": mlflow_run_id,
                    "timestamp": _now_iso(),
                }
            },
        )

    def track_alerting(self, run_id: str, n_alerts: int) -> None:
        self.emit_complete(
            run_id=run_id,
            job_name="alerting.alerts",
            inputs=["data/processed/anomaly_scores.parquet"],
            outputs=["data/processed/alerts.jsonl"],
            row_counts={"alerts": n_alerts},
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_event(
        self,
        event_type: str,
        job_name: str,
        run_id: str,
        inputs: list[str],
        outputs: list[str],
        row_counts: dict[str, int] | None = None,
        extra_facets: dict | None = None,
    ) -> dict:
        rc = row_counts or {}

        def _ds(name: str, direction: str) -> dict:
            schema_key = name.split("/")[-1].replace(".parquet", "").replace(".jsonl", "")
            facets: dict[str, Any] = {}
            if schema_key in self.SCHEMAS:
                facets.update(_schema_facet(self.SCHEMAS[schema_key]))
            if schema_key in rc:
                facets.update(_data_quality_facet(rc[schema_key]))
            return _dataset(self.NAMESPACE, name, facets or None)

        event: dict[str, Any] = {
            "eventType": event_type,
            "eventTime": _now_iso(),
            "producer": "https://github.com/yourusername/sensorops",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
            "run": {
                "runId": run_id,
                "facets": extra_facets or {},
            },
            "job": {
                "namespace": self.NAMESPACE,
                "name": job_name,
                "facets": {},
            },
            "inputs": [_ds(i, "input") for i in inputs],
            "outputs": [_ds(o, "output") for o in outputs],
        }
        return event

    def _send(self, event: dict) -> None:
        """Send to OL server if configured, otherwise write to fallback JSONL."""
        if self._ol_client:
            try:
                self._ol_client.emit(event)
                return
            except Exception as exc:
                print(f"[lineage] OL server emit failed ({exc}), falling back to file")

        # Fallback: append to JSONL
        self._fallback_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self._fallback_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
