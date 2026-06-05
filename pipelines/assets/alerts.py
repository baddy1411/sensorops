"""
Asset: alerts

Filters anomaly_scores for flagged events and emits structured alert records.

Each alert contains:
  - alert_id        : UUID
  - machine_id      : which unit triggered
  - timestamp       : event time
  - anomaly_score   : model score [0, 1]
  - failure_type    : from the dataset label (or UNKNOWN for real-time)
  - severity        : LOW / MEDIUM / HIGH / CRITICAL  (score-based)
  - top_features    : which features deviated most (simple z-score ranking)
  - raw_event       : full original sensor reading

Sink types (via AlertSinkResource):
  console  → logs each alert via context.log.warning
  file     → appends JSON lines to output_path
  webhook  → HTTP POST to webhook_url (e.g. Slack)

Dagster asset key : alerts
Input             : anomaly_scores (DataFrame)
Output            : alert records  (also written to sink)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

import httpx
import pandas as pd
from dagster import asset
from pydantic import BaseModel

from pipelines.resources import AlertSinkResource

SeverityLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Score → severity bands
_SEVERITY_BANDS: list[tuple[float, SeverityLevel]] = [
    (0.90, "CRITICAL"),
    (0.75, "HIGH"),
    (0.60, "MEDIUM"),
    (0.0, "LOW"),
]

# Feature columns used for deviation ranking
_NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "vibration_ms2",
    "temp_delta",
    "power_proxy_kw",
    "vibration_energy",
    "wear_ratio",
]


class Alert(BaseModel):
    alert_id: str
    machine_id: str
    timestamp: str
    anomaly_score: float
    severity: SeverityLevel
    failure_type: str
    top_features: list[str]
    sensor_snapshot: dict


def _severity(score: float) -> SeverityLevel:
    for threshold, level in _SEVERITY_BANDS:
        if score >= threshold:
            return level
    return "LOW"


def _top_deviating_features(row: pd.Series, df: pd.DataFrame, n: int = 3) -> list[str]:
    """Return the n features with highest absolute z-score deviation for this row."""
    avail = [c for c in _NUMERIC_FEATURES if c in df.columns]
    if not avail:
        return []
    means = df[avail].mean()
    stds = df[avail].std().replace(0, 1)
    z_scores = ((row[avail] - means) / stds).abs().astype(float)
    return z_scores.nlargest(n).index.tolist()


def _build_alert(row: pd.Series, df: pd.DataFrame) -> Alert:
    return Alert(
        alert_id=str(uuid.uuid4()),
        machine_id=str(row.get("machine_id", "UNKNOWN")),
        timestamp=str(row.get("timestamp", "")),
        anomaly_score=round(float(row["anomaly_score"]), 4),
        severity=_severity(float(row["anomaly_score"])),
        failure_type=str(row.get("failure_type", "UNKNOWN")),
        top_features=_top_deviating_features(row, df),
        sensor_snapshot={
            "air_temperature_k": row.get("air_temperature_k"),
            "process_temperature_k": row.get("process_temperature_k"),
            "rotational_speed_rpm": row.get("rotational_speed_rpm"),
            "torque_nm": row.get("torque_nm"),
            "tool_wear_min": row.get("tool_wear_min"),
            "vibration_ms2": row.get("vibration_ms2"),
        },
    )


def _emit_console(alerts: list[Alert], context) -> None:
    for a in alerts:
        context.log.warning(
            f"[ALERT {a.severity}] machine={a.machine_id} "
            f"score={a.anomaly_score:.3f} "
            f"failure_type={a.failure_type} "
            f"top_features={a.top_features}"
        )


def _emit_file(alerts: list[Alert], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for a in alerts:
            f.write(a.model_dump_json() + "\n")


def _emit_webhook(alerts: list[Alert], url: str) -> None:
    payload = {"alerts": [a.model_dump() for a in alerts]}
    try:
        resp = httpx.post(url, json=payload, timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        # Don't crash the pipeline on webhook failure — log and continue
        print(f"[alerts] webhook POST failed: {exc}")


@asset(
    group_name="alerting",
    description="Structured anomaly alerts emitted to configured sink (console/file/webhook).",
)
def alerts(
    context,
    anomaly_scores: pd.DataFrame,
    alert_sink: AlertSinkResource,
) -> list[dict]:
    """
    Filter flagged events, build structured Alert records, and emit to sink.
    Returns list of alert dicts for downstream assets (e.g. LLM report generator).
    """
    flagged = anomaly_scores[anomaly_scores["is_anomaly"] == 1].copy()

    if flagged.empty:
        context.log.info("alerts: no anomalies detected in this batch")
        return []

    alert_records = [_build_alert(row, anomaly_scores) for _, row in flagged.iterrows()]

    # Emit to configured sink
    sink = alert_sink.sink_type
    if sink == "console":
        _emit_console(alert_records, context)
    elif sink == "file":
        _emit_file(alert_records, alert_sink.output_path)
    elif sink == "webhook":
        _emit_webhook(alert_records, alert_sink.webhook_url)
    else:
        context.log.warning(f"Unknown sink type '{sink}' — defaulting to console")
        _emit_console(alert_records, context)

    severity_counts = {}
    for a in alert_records:
        severity_counts[a.severity] = severity_counts.get(a.severity, 0) + 1

    context.add_output_metadata(
        {
            "total_alerts": len(alert_records),
            "severity_breakdown": severity_counts,
            "sink_type": sink,
        }
    )

    context.log.info(
        f"alerts: {len(alert_records)} alerts emitted "
        f"| severity breakdown: {severity_counts}"
    )

    return [a.model_dump() for a in alert_records]
