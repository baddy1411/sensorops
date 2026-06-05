"""
Incident Reporter — anomaly detected → LLM generates structured incident report.

Pipeline:
  1. Receive alert dict from the alerts asset
  2. Retrieve relevant failure mode + manual context from RAG
  3. Call DeepSeek to generate a structured JSON incident report
  4. Validate and return the report as a typed Pydantic model
  5. (Optional) Write to audit log

The JSON schema is enforced via Pydantic — if the LLM returns malformed JSON
we retry once with an explicit correction prompt before raising.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from llm.prompts import SYSTEM_INCIDENT_REPORTER, build_incident_report_prompt
from llm.rag import RAGStore

# ---------------------------------------------------------------------------
# Typed report schema
# ---------------------------------------------------------------------------


class RecommendedAction(BaseModel):
    priority: int
    action: str
    timeframe: Literal["immediate", "within 4h", "within 24h", "within 1 week"]


class IncidentReport(BaseModel):
    incident_id: str
    machine_id: str
    timestamp: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    probable_cause: str
    evidence: list[str]
    recommended_actions: list[RecommendedAction] = Field(min_length=1)
    affected_components: list[str]
    model_version: str
    dataset_hash: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    notes: str = ""


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class IncidentReporter:
    """
    Generates structured incident reports from anomaly alerts using DeepSeek.

    Usage:
        reporter = IncidentReporter(rag_store=store)
        report = reporter.generate(alert_dict, model_version="IF-v1.2")
        print(report.probable_cause)
    """

    def __init__(
        self,
        rag_store: RAGStore,
        model: str = "deepseek-chat",
        max_tokens: int = 2048,
        api_key: str | None = None,
        dataset_hash: str | None = None,
    ) -> None:
        self.rag_store = rag_store
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        # Dataset hash for EU AI Act audit trail
        self._dataset_hash = dataset_hash or self._compute_default_hash()

    def generate(
        self,
        alert: dict,
        model_version: str = "unknown",
        n_context_docs: int = 4,
    ) -> IncidentReport:
        """
        Generate a structured incident report for an alert.

        Retries once on JSON parse failure with a correction prompt.
        """
        context_docs = self.rag_store.retrieve(
            query=(
                f"{alert.get('failure_type', '')} failure "
                f"machine {alert.get('machine_id', '')} "
                f"anomaly features {' '.join(alert.get('top_features', []))}"
            ),
            n_results=n_context_docs,
        )

        prompt = build_incident_report_prompt(
            alert=alert,
            retrieved_context=context_docs,
            model_version=model_version,
            dataset_hash=self._dataset_hash,
        )

        raw_json = self._call_llm(prompt)
        return self._parse_report(raw_json, alert, model_version)

    def generate_batch(
        self,
        alerts: list[dict],
        model_version: str = "unknown",
    ) -> list[IncidentReport]:
        """Generate reports for multiple alerts. Returns list (same order)."""
        return [self.generate(a, model_version=model_version) for a in alerts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_INCIDENT_REPORTER},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _parse_report(self, raw_json: str, alert: dict, model_version: str) -> IncidentReport:
        """Parse Claude's JSON output into a validated IncidentReport."""
        # Strip accidental markdown fences if present
        text = raw_json
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Retry with explicit correction
            correction_prompt = (
                f"The following text is not valid JSON. "
                f"Return only the corrected JSON, no markdown:\n\n{text}"
            )
            corrected = self._call_llm(correction_prompt)
            data = json.loads(corrected)

        # Fill in required audit fields if Claude omitted them
        data.setdefault("incident_id", alert.get("alert_id", "unknown"))
        data.setdefault("machine_id", alert.get("machine_id", "UNKNOWN"))
        data.setdefault("timestamp", alert.get("timestamp", ""))
        data.setdefault("model_version", model_version)
        data.setdefault("dataset_hash", self._dataset_hash)

        return IncidentReport.model_validate(data)

    @staticmethod
    def _compute_default_hash() -> str:
        """Placeholder hash — in production computed from actual dataset file."""
        csv_path = "data/raw/ai4i2020.csv"
        try:
            import pathlib

            content = pathlib.Path(csv_path).read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        except FileNotFoundError:
            return "hash_unavailable"
