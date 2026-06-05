"""
RAG store for SensorOps — ChromaDB-backed retrieval.

Stores and retrieves:
  - Maintenance manual chunks (indexed at startup from data/manuals/)
  - Historical alert records (indexed as they are emitted)
  - Failure mode descriptions (AI4I failure subtypes: TWF, HDF, PWF, OSF, RNF)

Embedding strategy:
  Uses ChromaDB's default embedding function (all-MiniLM-L6-v2 via sentence-transformers)
  for local operation with no API calls on the embedding side.

Collections:
  sensorops_manuals   — static maintenance knowledge
  sensorops_alerts    — live alert history
  sensorops_failures  — failure mode reference cards
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# Built-in failure mode knowledge — indexed at startup so the LLM always
# has structured context about what each failure subtype means.
FAILURE_MODE_CARDS = {
    "TWF": {
        "name": "Tool Wear Failure",
        "description": (
            "Occurs when tool wear exceeds the replacement threshold (typically 200–250 min). "
            "Characterised by gradual degradation of surface finish and dimensional accuracy. "
            "Early indicators: increasing torque at constant RPM, rising vibration harmonics."
        ),
        "typical_features": "tool_wear_min > 200, torque_nm elevated, vibration harmonics",
        "recommended_action": "Schedule tool replacement. Inspect workpiece quality.",
    },
    "HDF": {
        "name": "Heat Dissipation Failure",
        "description": (
            "Triggered when the temperature difference (process_temp - air_temp) falls below "
            "8.6 K at rotational speeds < 1380 rpm. Indicates insufficient cooling. "
            "Risk of thermal damage to bearing and spindle components."
        ),
        "typical_features": "temp_delta < 8.6 K, rotational_speed_rpm < 1380",
        "recommended_action": "Check coolant flow rate. Inspect heat exchanger. Reduce feed rate.",
    },
    "PWF": {
        "name": "Power Failure",
        "description": (
            "Power outside the operating range [3500, 9000] W. "
            "Computed as torque × angular_velocity. "
            "Under-power: motor stall risk. Over-power: thermal overload of drive."
        ),
        "typical_features": "power_proxy_kw outside [3.5, 9.0] kW",
        "recommended_action": "Check motor drive parameters. Verify load conditions.",
    },
    "OSF": {
        "name": "Overstrain Failure",
        "description": (
            "Product of tool wear and torque exceeds strain threshold. "
            "Thresholds differ by product quality: L=11,380, M=12,240, H=13,000 Nm·min. "
            "Leads to catastrophic tool breakage if not caught early."
        ),
        "typical_features": "tool_wear_min × torque_nm > quality-dependent threshold",
        "recommended_action": "Immediate tool inspection. Reduce cutting depth. Replace tool.",
    },
    "RNF": {
        "name": "Random Failure",
        "description": (
            "Rare, unpredictable failure with probability 0.1% per sample. "
            "No single dominant feature signature. "
            "May indicate bearing spall, foreign object, or sensor fault."
        ),
        "typical_features": "No consistent pattern — broad anomaly across multiple sensors",
        "recommended_action": (
            "Full machine inspection. Check for foreign objects and loose components."
        ),
    },
}


class RAGStore:
    """
    ChromaDB-backed retrieval store for SensorOps.

    Usage:
        store = RAGStore(persist_dir="data/chroma")
        store.index_failure_modes()           # index built-in knowledge
        store.index_manual_directory("data/manuals/")
        docs = store.retrieve("why did unit 3 overheat?", n_results=4)
    """

    def __init__(self, persist_dir: str = "data/chroma") -> None:
        self.persist_dir = persist_dir
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = embedding_functions.DefaultEmbeddingFunction()

        self._manuals = self._client.get_or_create_collection(
            name="sensorops_manuals",
            embedding_function=self._ef,
            metadata={"description": "Maintenance manual chunks"},
        )
        self._alerts = self._client.get_or_create_collection(
            name="sensorops_alerts",
            embedding_function=self._ef,
            metadata={"description": "Historical alert records"},
        )
        self._failures = self._client.get_or_create_collection(
            name="sensorops_failures",
            embedding_function=self._ef,
            metadata={"description": "Failure mode reference cards"},
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_failure_modes(self) -> int:
        """Index the built-in failure mode cards. Safe to call multiple times."""
        existing = set(self._failures.get()["ids"])
        new_docs, new_ids, new_metas = [], [], []

        for code, card in FAILURE_MODE_CARDS.items():
            doc_id = f"failure_mode_{code}"
            if doc_id in existing:
                continue
            text = (
                f"Failure type: {code} — {card['name']}\n"
                f"Description: {card['description']}\n"
                f"Typical features: {card['typical_features']}\n"
                f"Recommended action: {card['recommended_action']}"
            )
            new_docs.append(text)
            new_ids.append(doc_id)
            new_metas.append({"source": "built_in", "failure_type": code})

        if new_docs:
            self._failures.add(documents=new_docs, ids=new_ids, metadatas=new_metas)
        return len(new_docs)

    def index_manual_directory(self, manual_dir: str, chunk_size: int = 500) -> int:
        """
        Chunk and index all .txt and .md files in manual_dir.
        Returns number of new chunks indexed.
        """
        manual_path = Path(manual_dir)
        if not manual_path.exists():
            return 0

        existing = set(self._manuals.get()["ids"])
        total_indexed = 0

        for file in [*manual_path.glob("**/*.txt"), *manual_path.glob("**/*.md")]:
            text = file.read_text(encoding="utf-8", errors="ignore")
            chunks = _chunk_text(text, chunk_size)

            for i, chunk in enumerate(chunks):
                doc_id = _doc_id(str(file), i)
                if doc_id in existing:
                    continue
                self._manuals.add(
                    documents=[chunk],
                    ids=[doc_id],
                    metadatas=[{"source": str(file), "chunk": i}],
                )
                total_indexed += 1

        return total_indexed

    def index_alert(self, alert: dict) -> None:
        """Index a single alert dict into the alerts collection."""
        doc_id = alert.get("alert_id", _doc_id(json.dumps(alert), 0))
        text = (
            f"Alert for machine {alert.get('machine_id', 'UNKNOWN')} "
            f"at {alert.get('timestamp', 'unknown time')}. "
            f"Anomaly score: {alert.get('anomaly_score', 0):.3f}. "
            f"Severity: {alert.get('severity', 'UNKNOWN')}. "
            f"Failure type: {alert.get('failure_type', 'UNKNOWN')}. "
            f"Top deviating features: {', '.join(alert.get('top_features', []))}. "
            f"Sensor snapshot: {json.dumps(alert.get('sensor_snapshot', {}))}"
        )
        self._alerts.upsert(
            documents=[text],
            ids=[doc_id],
            metadatas=[
                {
                    "machine_id": str(alert.get("machine_id", "")),
                    "severity": str(alert.get("severity", "")),
                    "failure_type": str(alert.get("failure_type", "")),
                }
            ],
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        collections: list[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant documents across all (or specified) collections.

        Returns a list of dicts with keys: text, source, collection, distance.
        """
        if collections is None:
            collections = ["failures", "manuals", "alerts"]

        results: list[dict] = []

        collection_map = {
            "failures": self._failures,
            "manuals": self._manuals,
            "alerts": self._alerts,
        }

        for name in collections:
            col = collection_map.get(name)
            if col is None or col.count() == 0:
                continue
            k = min(n_results, col.count())
            res = col.query(query_texts=[query], n_results=k)
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                results.append(
                    {
                        "text": doc,
                        "source": meta.get("source", name),
                        "collection": name,
                        "distance": dist,
                    }
                )

        # Sort by distance (lower = more relevant)
        results.sort(key=lambda x: x["distance"])
        return results[:n_results]

    def retrieve_for_machine(self, machine_id: str, n_results: int = 5) -> list[dict]:
        """Retrieve alerts specifically for a given machine."""
        if self._alerts.count() == 0:
            return []
        k = min(n_results, self._alerts.count())
        res = self._alerts.query(
            query_texts=[f"machine {machine_id} failure anomaly"],
            n_results=k,
            where={"machine_id": machine_id} if machine_id else None,
        )
        return [
            {"text": doc, "source": "alerts", "collection": "alerts", "distance": dist}
            for doc, dist in zip(res["documents"][0], res["distances"][0])
        ]

    @property
    def stats(self) -> dict:
        return {
            "manuals": self._manuals.count(),
            "alerts": self._alerts.count(),
            "failures": self._failures.count(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    words = text.split()
    chunks, current = [], []
    length = 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            # 20% overlap
            overlap = max(1, len(current) // 5)
            current = current[-overlap:]
            length = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _doc_id(source: str, index: int) -> str:
    h = hashlib.md5(f"{source}:{index}".encode()).hexdigest()[:12]
    return f"doc_{h}"
