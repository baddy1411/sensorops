"""
EU AI Act Audit Logger for SensorOps.

EU AI Act (Article 12) requires high-risk AI systems to maintain logs of:
  - Model identity (name, version)
  - Training data provenance (dataset hash)
  - Each prediction: input data, output, timestamp
  - Human oversight events (model promotions, alert acknowledgements)

This logger writes append-only JSONL audit records. Records are:
  - Immutable once written (append-only file)
  - Hashed for tamper-evidence (each record includes SHA256 of previous record)
  - Structured for export to regulatory bodies

Log format (one JSON object per line):
  {
    "seq":           <monotonic sequence number>,
    "event_type":    <PREDICTION|PROMOTION|ALERT|HUMAN_OVERSIGHT>,
    "timestamp":     <ISO 8601 UTC>,
    "model_name":    <string>,
    "model_version": <string>,
    "dataset_hash":  <sha256 hex>,
    "payload":       <event-specific data>,
    "prev_hash":     <sha256 of previous record — chain integrity>,
    "record_hash":   <sha256 of this record>
  }
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

EventType = Literal["PREDICTION", "PROMOTION", "ALERT", "HUMAN_OVERSIGHT", "PIPELINE_RUN"]


class AuditLogger:
    """
    Append-only, hash-chained EU AI Act audit log.

    Thread-safe. Designed to run in the same process as the model server
    or the Dagster pipeline.

    Usage:
        log = AuditLogger()
        log.log_prediction(
            model_name="isolation_forest",
            model_version="v1.2",
            dataset_hash="abc123",
            machine_id="M14860",
            anomaly_score=0.82,
            is_anomaly=True,
        )
    """

    def __init__(
        self,
        log_path: str = "data/processed/audit.jsonl",
        model_name: str = "unknown",
        model_version: str = "unknown",
        dataset_hash: str = "unknown",
    ) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._model_version = model_version
        self._dataset_hash = dataset_hash
        self._lock = threading.Lock()
        self._seq = self._load_seq()
        self._prev_hash = self._load_last_hash()

    # ------------------------------------------------------------------
    # Public logging methods
    # ------------------------------------------------------------------

    def log_prediction(
        self,
        machine_id: str,
        anomaly_score: float,
        is_anomaly: bool,
        input_features: dict | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
    ) -> str:
        """Log one model prediction. Returns the record hash."""
        return self._write(
            event_type="PREDICTION",
            model_name=model_name or self._model_name,
            model_version=model_version or self._model_version,
            payload={
                "machine_id": machine_id,
                "anomaly_score": round(anomaly_score, 6),
                "is_anomaly": is_anomaly,
                "input_features": input_features or {},
            },
        )

    def log_alert(
        self,
        alert_id: str,
        machine_id: str,
        severity: str,
        failure_type: str,
        anomaly_score: float,
    ) -> str:
        """Log an anomaly alert emission."""
        return self._write(
            event_type="ALERT",
            model_name=self._model_name,
            model_version=self._model_version,
            payload={
                "alert_id": alert_id,
                "machine_id": machine_id,
                "severity": severity,
                "failure_type": failure_type,
                "anomaly_score": round(anomaly_score, 6),
            },
        )

    def log_promotion(
        self,
        model_name: str,
        from_stage: str,
        to_stage: str,
        model_version: str,
        triggered_by: str,
        metrics: dict,
    ) -> str:
        """Log a model registry stage transition (Staging → Production)."""
        return self._write(
            event_type="PROMOTION",
            model_name=model_name,
            model_version=model_version,
            payload={
                "from_stage": from_stage,
                "to_stage": to_stage,
                "triggered_by": triggered_by,
                "metrics": metrics,
                "requires_human_approval": to_stage == "Production",
            },
        )

    def log_human_oversight(
        self,
        action: str,
        operator_id: str,
        alert_id: str | None = None,
        notes: str = "",
    ) -> str:
        """Log a human oversight event (operator acknowledged alert, overrode model, etc.)."""
        return self._write(
            event_type="HUMAN_OVERSIGHT",
            model_name=self._model_name,
            model_version=self._model_version,
            payload={
                "action": action,
                "operator_id": operator_id,
                "alert_id": alert_id,
                "notes": notes,
            },
        )

    def log_pipeline_run(
        self,
        job_name: str,
        run_id: str,
        status: Literal["started", "completed", "failed"],
        row_count: int = 0,
        duration_s: float = 0.0,
        error: str | None = None,
    ) -> str:
        """Log a Dagster pipeline run boundary."""
        return self._write(
            event_type="PIPELINE_RUN",
            model_name=self._model_name,
            model_version=self._model_version,
            payload={
                "job_name": job_name,
                "run_id": run_id,
                "status": status,
                "row_count": row_count,
                "duration_s": round(duration_s, 3),
                "error": error,
            },
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain(self) -> tuple[bool, list[str]]:
        """
        Verify the hash chain integrity of the audit log.

        Returns (is_valid, list_of_errors).
        An empty error list means the log is untampered.
        """
        errors: list[str] = []
        prev_hash = ""

        with open(self._log_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"Line {line_no}: invalid JSON")
                    continue

                # Check prev_hash linkage
                if record.get("prev_hash") != prev_hash:
                    errors.append(
                        f"Line {line_no} (seq={record.get('seq')}): "
                        f"prev_hash mismatch — chain broken"
                    )

                # Recompute record_hash
                stored_hash = record.pop("record_hash", "")
                recomputed = hashlib.sha256(
                    json.dumps(record, sort_keys=True).encode()
                ).hexdigest()
                if recomputed != stored_hash:
                    errors.append(
                        f"Line {line_no} (seq={record.get('seq')}): "
                        f"record_hash mismatch — record may have been tampered"
                    )

                prev_hash = stored_hash

        return len(errors) == 0, errors

    def get_stats(self) -> dict:
        """Return counts by event type."""
        if not self._log_path.exists():
            return {}
        counts: dict[str, int] = {}
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    et = rec.get("event_type", "UNKNOWN")
                    counts[et] = counts.get(et, 0) + 1
                except json.JSONDecodeError:
                    pass
        return counts

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write(
        self,
        event_type: EventType,
        model_name: str,
        model_version: str,
        payload: dict[str, Any],
    ) -> str:
        with self._lock:
            self._seq += 1
            record: dict[str, Any] = {
                "seq": self._seq,
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": model_name,
                "model_version": model_version,
                "dataset_hash": self._dataset_hash,
                "payload": payload,
                "prev_hash": self._prev_hash,
            }

            record_hash = hashlib.sha256(
                json.dumps(record, sort_keys=True).encode()
            ).hexdigest()
            record["record_hash"] = record_hash

            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            self._prev_hash = record_hash
            return record_hash

    def _load_seq(self) -> int:
        if not self._log_path.exists():
            return 0
        last_seq = 0
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    last_seq = json.loads(line).get("seq", last_seq)
                except json.JSONDecodeError:
                    pass
        return last_seq

    def _load_last_hash(self) -> str:
        if not self._log_path.exists():
            return ""
        last_hash = ""
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    last_hash = json.loads(line).get("record_hash", last_hash)
                except json.JSONDecodeError:
                    pass
        return last_hash
