"""
System and user prompt templates for all SensorOps LLM interactions.

Keeping prompts in one file makes them easy to iterate, version, and test
independently from the API/agent logic.
"""

from __future__ import annotations

SYSTEM_QUERY_ENGINE = """\
You are SensorOps Assistant, an expert industrial AI system for predictive maintenance
on manufacturing equipment (Industrie 4.0 context, German automotive sector).

You have access to:
- Real-time sensor telemetry (air temp, process temp, RPM, torque, tool wear, vibration)
- Anomaly detection model outputs (Isolation Forest, LSTM Autoencoder, Echo State Network)
- Maintenance manual knowledge
- Historical alert records

Failure mode reference:
- TWF (Tool Wear Failure): tool_wear_min > 200, elevated torque + vibration harmonics
- HDF (Heat Dissipation Failure): temp_delta < 8.6 K at RPM < 1380
- PWF (Power Failure): power outside [3.5, 9.0] kW
- OSF (Overstrain Failure): tool_wear × torque exceeds quality-tier threshold
- RNF (Random Failure): low-probability unpredictable failure, no single signature

Guidelines:
- Be precise and technical. Cite specific sensor values when available.
- Distinguish clearly between confirmed failures and anomaly model flags.
- If asked why a machine failed, reason from the sensor snapshot + failure mode definitions.
- Always recommend a concrete next action (inspect, replace, reduce feed rate, etc.).
- If context is insufficient, say so — do not hallucinate sensor readings.
- Keep responses concise: operators need answers fast.
"""


SYSTEM_INCIDENT_REPORTER = """\
You are an industrial incident reporting system for Industrie 4.0 manufacturing.
You generate structured, professional incident reports from sensor anomaly data.

Your reports must be:
- Factual: based only on provided sensor data and known failure mode definitions
- Structured: follow the exact JSON schema provided
- Actionable: every report includes a prioritised action list
- Compliant: include model version and data hash for EU AI Act audit trail
- Concise: engineers read these under time pressure
"""


def build_query_prompt(
    user_question: str,
    retrieved_context: list[dict],
    recent_alerts: list[dict] | None = None,
) -> str:
    """Build the user-turn prompt for the NL query engine."""
    ctx_blocks = []
    for i, doc in enumerate(retrieved_context, 1):
        ctx_blocks.append(f"[Context {i} — {doc['collection']}]\n{doc['text']}")

    context_section = (
        "\n\n".join(ctx_blocks) if ctx_blocks else "No relevant context found in knowledge base."
    )

    alert_section = ""
    if recent_alerts:
        alert_lines = []
        for a in recent_alerts[:5]:
            alert_lines.append(
                f"  - Machine {a.get('machine_id')} | "
                f"Score {a.get('anomaly_score', 0):.3f} | "
                f"Severity {a.get('severity')} | "
                f"Type {a.get('failure_type')}"
            )
        alert_section = "\n\nRecent alerts:\n" + "\n".join(alert_lines)

    return f"""\
Question: {user_question}

Relevant context from knowledge base:
{context_section}{alert_section}

Answer the question using the context above. Be specific and actionable."""


def build_incident_report_prompt(
    alert: dict,
    retrieved_context: list[dict],
    model_version: str,
    dataset_hash: str,
) -> str:
    """Build the prompt that asks Claude to generate a structured incident report."""
    ctx_text = "\n\n".join(f"[{doc['collection']}] {doc['text']}" for doc in retrieved_context[:4])

    sensor = alert.get("sensor_snapshot", {})
    sensor_str = "\n".join(f"  {k}: {v}" for k, v in sensor.items())

    return f"""\
Generate a structured incident report for the following anomaly detection event.

=== ALERT DATA ===
Machine ID:       {alert.get("machine_id", "UNKNOWN")}
Timestamp:        {alert.get("timestamp", "UNKNOWN")}
Anomaly Score:    {alert.get("anomaly_score", 0):.4f}
Severity:         {alert.get("severity", "UNKNOWN")}
Failure Type:     {alert.get("failure_type", "UNKNOWN")}
Top Features:     {", ".join(alert.get("top_features", []))}

Sensor Snapshot:
{sensor_str}

=== AUDIT METADATA ===
Model Version:    {model_version}
Dataset Hash:     {dataset_hash}

=== KNOWLEDGE BASE CONTEXT ===
{ctx_text if ctx_text else "No additional context available."}

=== OUTPUT FORMAT ===
Return a JSON object with exactly these fields:
{{
  "incident_id": "<same as alert_id if available>",
  "machine_id": "<machine id>",
  "timestamp": "<ISO timestamp>",
  "severity": "<LOW|MEDIUM|HIGH|CRITICAL>",
  "probable_cause": "<1-2 sentence root cause analysis>",
  "evidence": ["<specific sensor reading or pattern that supports the diagnosis>", ...],
  "recommended_actions": [
    {{"priority": 1, "action": "<most urgent action>",  # noqa: E501
     "timeframe": "<immediate|within 4h|within 24h>"}},
    ...
  ],
  "affected_components": ["<component name>", ...],
  "model_version": "{model_version}",
  "dataset_hash": "{dataset_hash}",
  "confidence": "<HIGH|MEDIUM|LOW>",
  "notes": "<any caveats or additional observations>"
}}

Return only valid JSON. No markdown fences."""
