"""
Dagster resource wrapping LineageEmitter + AuditLogger.

Inject as "lineage" resource into any asset that needs to emit
OpenLineage events or write EU AI Act audit records.
"""

from __future__ import annotations

import os

from dagster import ConfigurableResource
from pydantic import Field

from lineage.audit_log import AuditLogger
from lineage.emitter import LineageEmitter


class LineageResource(ConfigurableResource):
    """Combined OpenLineage + EU AI Act audit resource."""

    openlineage_url: str = Field(
        default="",
        description="OpenLineage / Marquez server URL. Empty = file fallback.",
    )
    lineage_log_path: str = Field(
        default="data/processed/lineage.jsonl",
        description="Fallback JSONL path for OL events.",
    )
    audit_log_path: str = Field(
        default="data/processed/audit.jsonl",
        description="EU AI Act audit log path.",
    )
    model_name: str = Field(default="isolation_forest")
    model_version: str = Field(default="unknown")
    dataset_hash: str = Field(default="unknown")

    def get_emitter(self) -> LineageEmitter:
        return LineageEmitter(
            url=self.openlineage_url or os.getenv("OPENLINEAGE_URL", ""),
            fallback_log=self.lineage_log_path,
        )

    def get_audit_logger(self) -> AuditLogger:
        return AuditLogger(
            log_path=self.audit_log_path,
            model_name=self.model_name,
            model_version=self.model_version,
            dataset_hash=self.dataset_hash,
        )
