<div align="center">

# ⚙️ SensorOps

### Industrial AI Platform for Predictive Maintenance

*Industrie 4.0 · MLOps · LLM Integration · EU AI Act Compliance*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Dagster](https://img.shields.io/badge/Dagster-1.13-purple?style=flat-square)](https://dagster.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-blue?style=flat-square)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-V3-4353ff?style=flat-square)](https://platform.deepseek.com)
[![Tests](https://img.shields.io/badge/Tests-90%20passing-3fb950?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](#)

<br/>

> **SensorOps** is a production-grade predictive maintenance platform that ingests real-time sensor telemetry, detects anomalies using three ML models (Isolation Forest, LSTM Autoencoder, Echo State Network), and uses a DeepSeek-powered LLM agent to answer natural language questions like *"Why did machine M14860 trigger a HIGH severity alert?"*

</div>

---

## 📐 Architecture

![Architecture](docs/images/architecture.png)

The platform is built in four independent layers that can each be deployed and scaled separately:

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Ingest** | CSV Replay Adapter, Pydantic schema | Stream sensor events with synthetic vibration channel |
| **MLOps Core** | Dagster, MLflow, FastAPI, Docker | Orchestrate, track, serve, and version models |
| **LLM Intelligence** | DeepSeek API, ChromaDB RAG | Answer NL queries, generate structured incident reports |
| **Compliance** | OpenLineage, EU AI Act audit log | Full data lineage + tamper-evident prediction log |

---

## 📊 Dataset & Signals

<table>
<tr>
<td width="60%">

**Primary dataset: [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601)**
- 10,000 machine telemetry snapshots
- ~3.5% failure rate (imbalanced — handled via contamination param)
- 5 failure subtypes: TWF · HDF · PWF · OSF · RNF
- Features: air temp, process temp, RPM, torque, tool wear

**Synthetic vibration channel** added on top:
- Physics-based sinusoid at running speed frequency
- 3× harmonic (bearing defect frequency)
- Gaussian noise floor σ = 0.05 m/s²
- Anomaly spike: 5–10× amplitude on failure events

</td>
<td width="40%">

| Feature | Range | Unit |
|---------|-------|------|
| Air temperature | 295–305 | K |
| Process temperature | 306–315 | K |
| Rotational speed | 1168–2886 | rpm |
| Torque | 3.8–76.6 | Nm |
| Tool wear | 0–253 | min |
| Vibration | synthetic | m/s² |

</td>
</tr>
</table>

![Dataset Overview](docs/images/dataset_overview.png)

---

## 🔬 Synthetic Vibration Channel

One of the platform's differentiators is a physics-informed synthetic vibration signal added to every sensor event — enabling richer anomaly detection beyond the 5 original dataset features.

![Vibration Channel](docs/images/vibration_channel.png)

The signal is modelled as:

```
v(t) = A_wear · sin(2π · f_rpm · t)           ← 1× running speed
      + 0.08 · sin(2π · 3f_rpm · t)           ← 3× bearing defect harmonic
      + ε(0, 0.05)                             ← noise floor
      [+ 5–10× spike if machine_failure=True]  ← anomaly injection
```

where `A_wear` grows linearly with tool wear (0.2 → 0.5 m/s²) to model degradation.

---

## 🤖 Models

### Isolation Forest (baseline)
Fully unsupervised. Contamination parameter set to 3.5% to match the known failure rate.  
Trains in seconds, inference in microseconds. Scores calibrated to [0, 1] via min-max on training distribution.

### LSTM Autoencoder
Learns the normal operating envelope via sequence reconstruction. Anomaly score = MSE reconstruction error, normalised to [0, 1]. Sequence length configurable (default: 30 timesteps).

### Echo State Network *(research-backed)*
> **Thesis context:** The ESN model is directly motivated by the author's M.Sc. thesis on *Quantum Reservoir Computing vs classical Echo State Networks* for time-series forecasting (Hénon map benchmark).
>
> **Result:** Tuned ESN achieves NRMSE **0.0111** vs QRC's **0.0130** at matched computational resources — ESN wins on both accuracy and efficiency.

Anomaly scoring uses 1-step-ahead prediction error: the reservoir learns the normal attractor, and deviations indicate out-of-distribution behaviour.

![Anomaly Scoring](docs/images/anomaly_scoring.png)

---

## 💬 LLM Query Engine

Operators can ask natural language questions about machine health. The system retrieves relevant context from a ChromaDB RAG store (failure mode reference cards + maintenance manuals + alert history) and calls DeepSeek to generate grounded answers.

```bash
POST /api/v1/llm/query
{
  "question": "Why did machine M14860 trigger a HIGH severity alert?",
  "machine_id": "M14860"
}
```

```json
{
  "answer": "Machine M14860 triggered a HIGH severity alert due to a Heat Dissipation Failure (HDF). The temperature differential (process_temp - air_temp) dropped to 7.8 K, below the 8.6 K threshold, at a rotational speed of 1320 rpm (below the 1380 rpm minimum). This indicates insufficient cooling. Recommended: check coolant flow rate immediately and inspect the heat exchanger within 4 hours.",
  "context_sources": ["built_in", "built_in", "alerts"],
  "input_tokens": 847,
  "output_tokens": 112
}
```

### Incident Reports

Every anomaly automatically generates a structured incident report:

```bash
POST /api/v1/llm/report
```

```json
{
  "incident_id": "alert-f3a2b1",
  "machine_id": "M14860",
  "severity": "HIGH",
  "probable_cause": "Heat dissipation failure caused by insufficient coolant flow at low RPM.",
  "evidence": [
    "temp_delta = 7.8 K (threshold: 8.6 K)",
    "rotational_speed = 1320 rpm (threshold: 1380 rpm)"
  ],
  "recommended_actions": [
    {"priority": 1, "action": "Check coolant flow rate", "timeframe": "immediate"},
    {"priority": 2, "action": "Inspect heat exchanger", "timeframe": "within 4h"}
  ],
  "affected_components": ["coolant system", "spindle bearing"],
  "model_version": "IF-v1.2",
  "dataset_hash": "a3f9c2b1e4d7",
  "confidence": "HIGH"
}
```

---

## 🔒 EU AI Act Compliance

SensorOps implements Article 12-level audit logging with **hash-chained tamper detection**:

```
Record 1: { seq:1, event_type:PREDICTION, model_version:IF-v1, dataset_hash:a3f9..., prev_hash:"",        record_hash:"d4e7..." }
Record 2: { seq:2, event_type:ALERT,      model_version:IF-v1, dataset_hash:a3f9..., prev_hash:"d4e7...", record_hash:"b2a1..." }
Record 3: { seq:3, event_type:PROMOTION,  requires_human_approval:true,              prev_hash:"b2a1...", record_hash:"f9c3..." }
```

- Every prediction logs: model name, version, dataset SHA-256, input features, anomaly score
- Every alert logs: severity, failure type, machine ID
- Production promotion **requires human approval** — automated promotion is disabled by design
- `verify_chain()` detects any tampering in O(n) time

---

## 🏗️ Project Structure

```
sensorops/
├── data/
│   ├── adapter.py          ← CSV replay adapter (async generator + Kafka)
│   ├── schema.py           ← SensorEvent Pydantic model
│   └── vibration.py        ← Synthetic vibration channel generator
│
├── models/
│   ├── base.py             ← BaseAnomalyModel (fit/score/predict/evaluate)
│   ├── isolation_forest.py ← IF wrapper
│   ├── lstm_autoencoder.py ← LSTM-AE (PyTorch, optional)
│   ├── esn.py              ← ESN (reservoirpy, thesis-backed)
│   └── registry.py         ← MLflow promotion logic + quality gates
│
├── pipelines/
│   ├── assets/             ← raw_events → features → anomaly_scores → alerts
│   ├── definitions.py      ← Dagster Definitions (entry point)
│   ├── jobs.py             ← 3 jobs (full / ingest-only / score-and-alert)
│   ├── schedules.py        ← hourly + daily cron
│   └── sensors.py          ← file-watch sensor (triggers on CSV update)
│
├── serving/
│   ├── app.py              ← FastAPI app (predict / batch / health / reload)
│   ├── feature_pipeline.py ← Real-time feature engineering (no pandas)
│   ├── model_loader.py     ← Thread-safe singleton loader
│   └── Dockerfile          ← Slim Python 3.11, non-root, 2 workers
│
├── llm/
│   ├── rag.py              ← ChromaDB RAG store (3 collections)
│   ├── query_engine.py     ← DeepSeek NL query engine
│   ├── incident_reporter.py← Structured incident report generator
│   ├── prompts.py          ← All prompts in one place
│   └── api.py              ← FastAPI router (/query /report /stream)
│
├── lineage/
│   ├── emitter.py          ← OpenLineage event emitter
│   ├── audit_log.py        ← EU AI Act hash-chained audit log
│   └── dagster_resource.py ← Dagster-injectable resource
│
├── infra/
│   └── docker-compose.yml  ← Full stack (API+MLflow+Dagster+Postgres+MinIO)
│
└── tests/                  ← 90 tests, all pass offline
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/sensorops.git
cd sensorops
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add your DEEPSEEK_API_KEY
```

### 3. Download dataset

Download [AI4I 2020](https://archive.ics.uci.edu/dataset/601) and place at `data/raw/ai4i2020.csv`.

### 4. Run tests

```bash
pytest          # 90 tests, all offline, ~25s
```

### 5. Preview the stream

```bash
python -m data.adapter --csv data/raw/ai4i2020.csv --limit 5
```

### 6. Launch the full stack

```bash
# Serving API only
uvicorn serving.app:app --reload --port 8000
# → http://localhost:8000/docs

# Full stack (API + MLflow + Dagster + Postgres + MinIO)
docker compose -f infra/docker-compose.yml up
# → Dagster UI:  http://localhost:3000
# → MLflow UI:   http://localhost:5000
# → API docs:    http://localhost:8000/docs
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness + model status |
| `POST` | `/api/v1/predict` | Score a single sensor reading |
| `POST` | `/api/v1/predict/batch` | Score up to 1000 readings |
| `POST` | `/api/v1/model/reload` | Hot-swap model without restart |
| `GET` | `/api/v1/model/info` | Current model metadata + params |
| `POST` | `/api/v1/llm/query` | NL question → grounded answer |
| `POST` | `/api/v1/llm/query/stream` | SSE streaming variant |
| `POST` | `/api/v1/llm/report` | Alert → structured incident report |
| `GET` | `/api/v1/llm/rag/stats` | RAG collection document counts |

**Example — score a reading:**

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "reading": {
      "machine_id": "M14860",
      "air_temperature_k": 298.1,
      "process_temperature_k": 308.6,
      "rotational_speed_rpm": 1551,
      "torque_nm": 42.8,
      "tool_wear_min": 0,
      "vibration_ms2": 0.15
    }
  }'
```

```json
{
  "result": {
    "machine_id": "M14860",
    "anomaly_score": 0.1823,
    "is_anomaly": false,
    "severity": "LOW",
    "threshold": 0.6
  },
  "model_name": "isolation_forest",
  "api_version": "v1"
}
```

---

## 🧪 Testing

```
tests/
├── test_adapter.py    ← 10 tests  — CSV replay, schema validation, vibration physics
├── test_models.py     ← 13 tests  — IF, LSTM (skip if no torch), ESN (skip if no reservoirpy)
├── test_pipeline.py   ← 20 tests  — Dagster assets, feature engineering, alerting
├── test_serving.py    ← 14 tests  — FastAPI endpoints, batch, health, schemas
├── test_llm.py        ← 20 tests  — RAG, prompts, query engine, incident reporter (all mocked)
└── test_lineage.py    ← 13 tests  — OL events, EU AI Act chain, tamper detection
```

All 90 tests pass with no external services — Claude/DeepSeek API is mocked, no MLflow server needed, ChromaDB runs in-memory.

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Orchestration | [Dagster](https://dagster.io) |
| ML tracking | [MLflow](https://mlflow.org) |
| Serving | [FastAPI](https://fastapi.tiangolo.com) + [Docker](https://docker.com) |
| Models | scikit-learn · PyTorch · [reservoirpy](https://reservoirpy.readthedocs.io) |
| LLM | [DeepSeek API](https://platform.deepseek.com) (OpenAI-compatible) |
| RAG | [ChromaDB](https://www.trychroma.com) |
| Data lineage | [OpenLineage](https://openlineage.io) |
| CI/CD | GitHub Actions |
| Storage | MinIO (S3-compatible) · pgvector (Postgres) |

---

## 📎 Research Background

The **Echo State Network** model is backed by original research from the author's M.Sc. thesis:

> *"Quantum Reservoir Computing vs Classical Echo State Networks for Time-Series Forecasting"*  
> Benchmark: Hénon map attractor prediction  
> **Result: Tuned ESN (NRMSE 0.0111) outperforms QRC (NRMSE 0.0130) at matched resources**

This means the ESN choice in SensorOps has a legitimate empirical justification — it is not a toy model added for novelty. The reservoir implementation uses [reservoirpy](https://reservoirpy.readthedocs.io) with spectral radius ρ=0.9 (consistent with the thesis tuning).

---

<div align="center">

Built for the German manufacturing / Industrie 4.0 market  
Portfolio project demonstrating production MLOps + LLM integration

</div>
