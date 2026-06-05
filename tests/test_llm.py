"""
Tests for the LLM intelligence layer.

Claude API calls are mocked — tests run fully offline.
ChromaDB runs in-memory (no persist_dir) so no disk state.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from llm.incident_reporter import IncidentReport, IncidentReporter
from llm.prompts import build_incident_report_prompt, build_query_prompt
from llm.query_engine import QueryEngine, QueryResult
from llm.rag import FAILURE_MODE_CARDS, RAGStore, _chunk_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_store(tmp_path) -> RAGStore:
    """RAGStore with a temp ChromaDB directory."""
    store = RAGStore(persist_dir=str(tmp_path / "chroma"))
    store.index_failure_modes()
    return store


SAMPLE_ALERT = {
    "alert_id": "alert-001",
    "machine_id": "M14860",
    "timestamp": "2024-01-15T08:32:10Z",
    "anomaly_score": 0.87,
    "severity": "HIGH",
    "failure_type": "HDF",
    "top_features": ["temp_delta", "rotational_speed_rpm", "power_proxy_kw"],
    "sensor_snapshot": {
        "air_temperature_k": 298.1,
        "process_temperature_k": 305.9,
        "rotational_speed_rpm": 1320.0,
        "torque_nm": 55.2,
        "tool_wear_min": 180.0,
        "vibration_ms2": 0.42,
    },
}


# ---------------------------------------------------------------------------
# RAG store tests
# ---------------------------------------------------------------------------


class TestRAGStore:
    def test_failure_modes_indexed(self, in_memory_store):
        assert in_memory_store.stats["failures"] == len(FAILURE_MODE_CARDS)

    def test_index_failure_modes_idempotent(self, in_memory_store):
        before = in_memory_store.stats["failures"]
        in_memory_store.index_failure_modes()  # call again
        assert in_memory_store.stats["failures"] == before

    def test_retrieve_returns_results(self, in_memory_store):
        docs = in_memory_store.retrieve("heat dissipation failure temperature")
        assert len(docs) > 0
        assert all("text" in d for d in docs)

    def test_retrieve_hdf_top_result(self, in_memory_store):
        docs = in_memory_store.retrieve("heat dissipation temperature coolant", n_results=3)
        texts = " ".join(d["text"] for d in docs[:2])
        assert "HDF" in texts or "Heat Dissipation" in texts

    def test_index_and_retrieve_alert(self, in_memory_store):
        in_memory_store.index_alert(SAMPLE_ALERT)
        assert in_memory_store.stats["alerts"] == 1
        docs = in_memory_store.retrieve_for_machine("M14860")
        assert len(docs) > 0

    def test_index_manual_missing_dir_returns_zero(self, in_memory_store, tmp_path):
        n = in_memory_store.index_manual_directory(str(tmp_path / "nonexistent"))
        assert n == 0

    def test_index_manual_from_file(self, in_memory_store, tmp_path):
        manual = tmp_path / "manual.txt"
        manual.write_text(
            "This is a maintenance manual. Check coolant flow regularly. "
            "Replace tool every 200 minutes. " * 20
        )
        n = in_memory_store.index_manual_directory(str(tmp_path))
        assert n > 0
        assert in_memory_store.stats["manuals"] > 0


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = _chunk_text("hello world", 500)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        long = "word " * 500
        chunks = _chunk_text(long, 100)
        assert len(chunks) > 1

    def test_chunks_cover_all_content(self):
        text = "the quick brown fox jumps over the lazy dog " * 50
        chunks = _chunk_text(text, 200)
        combined = " ".join(chunks)
        assert "quick" in combined
        assert "lazy" in combined


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_query_prompt_contains_question(self):
        prompt = build_query_prompt(
            user_question="Why did unit 3 fail?",
            retrieved_context=[],
        )
        assert "Why did unit 3 fail?" in prompt

    def test_query_prompt_includes_context(self):
        ctx = [{"text": "HDF failure context here", "collection": "failures", "distance": 0.1}]
        prompt = build_query_prompt("question", ctx)
        assert "HDF failure context here" in prompt

    def test_incident_prompt_contains_machine_id(self):
        ctx = [{"text": "relevant info", "collection": "failures", "distance": 0.1}]
        prompt = build_incident_report_prompt(
            alert=SAMPLE_ALERT,
            retrieved_context=ctx,
            model_version="IF-v1",
            dataset_hash="abc123",
        )
        assert "M14860" in prompt
        assert "IF-v1" in prompt
        assert "abc123" in prompt

    def test_incident_prompt_requests_json(self):
        prompt = build_incident_report_prompt(SAMPLE_ALERT, [], "v1", "h1")
        assert "JSON" in prompt


# ---------------------------------------------------------------------------
# Query engine tests (Claude mocked)
# ---------------------------------------------------------------------------


def _mock_deepseek_response(text: str) -> MagicMock:
    """Build a mock OpenAI-compatible chat completion response."""
    msg = MagicMock()
    msg.choices = [MagicMock(message=MagicMock(content=text))]
    msg.model = "deepseek-chat"
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_cache_hit_tokens = 80
    usage.prompt_cache_miss_tokens = 20
    msg.usage = usage
    return msg


class TestQueryEngine:
    def test_query_returns_result(self, in_memory_store):
        with patch("llm.query_engine.OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_deepseek_response(
                "The machine overheated due to low coolant flow."
            )
            engine = QueryEngine(rag_store=in_memory_store, api_key="fake-key")
            result = engine.query("Why did M14860 overheat?")

        assert isinstance(result, QueryResult)
        assert "overheat" in result.answer.lower()
        assert result.input_tokens == 100

    def test_query_with_machine_id(self, in_memory_store):
        in_memory_store.index_alert(SAMPLE_ALERT)
        with patch("llm.query_engine.OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_deepseek_response(
                "Analysis done."
            )
            engine = QueryEngine(rag_store=in_memory_store, api_key="fake-key")
            result = engine.query("What happened?", machine_id="M14860")

        assert result.answer == "Analysis done."

    def test_query_context_sources_populated(self, in_memory_store):
        with patch("llm.query_engine.OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_deepseek_response("OK")
            engine = QueryEngine(rag_store=in_memory_store, api_key="fake-key")
            result = engine.query("tool wear failure")

        assert len(result.retrieved_context) > 0


# ---------------------------------------------------------------------------
# Incident reporter tests (Claude mocked)
# ---------------------------------------------------------------------------


VALID_REPORT_JSON = json.dumps(
    {
        "incident_id": "alert-001",
        "machine_id": "M14860",
        "timestamp": "2024-01-15T08:32:10Z",
        "severity": "HIGH",
        "probable_cause": "Insufficient heat dissipation at low RPM caused thermal stress.",
        "evidence": ["temp_delta=7.8 K (below 8.6 K threshold)", "RPM=1320 (below 1380 threshold)"],
        "recommended_actions": [
            {"priority": 1, "action": "Check coolant flow rate", "timeframe": "immediate"},
            {"priority": 2, "action": "Inspect heat exchanger", "timeframe": "within 4h"},
        ],
        "affected_components": ["coolant system", "spindle bearing"],
        "model_version": "IF-v1",
        "dataset_hash": "abc123",
        "confidence": "HIGH",
        "notes": "Pattern consistent with HDF failure mode.",
    }
)


class TestIncidentReporter:
    def test_generate_returns_incident_report(self, in_memory_store):
        with patch("llm.incident_reporter.OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_deepseek_response(
                VALID_REPORT_JSON
            )
            reporter = IncidentReporter(rag_store=in_memory_store, api_key="fake-key")
            report = reporter.generate(SAMPLE_ALERT, model_version="IF-v1")

        assert isinstance(report, IncidentReport)
        assert report.machine_id == "M14860"
        assert report.severity == "HIGH"
        assert len(report.recommended_actions) >= 1
        assert report.recommended_actions[0].priority == 1

    def test_report_has_audit_fields(self, in_memory_store):
        with patch("llm.incident_reporter.OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_deepseek_response(
                VALID_REPORT_JSON
            )
            reporter = IncidentReporter(rag_store=in_memory_store, api_key="fake-key")
            report = reporter.generate(SAMPLE_ALERT, model_version="IF-v1")

        assert report.model_version is not None
        assert report.dataset_hash is not None

    def test_report_strips_markdown_fences(self, in_memory_store):
        fenced = f"```json\n{VALID_REPORT_JSON}\n```"
        with patch("llm.incident_reporter.OpenAI") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat.completions.create.return_value = _mock_deepseek_response(fenced)
            reporter = IncidentReporter(rag_store=in_memory_store, api_key="fake-key")
            report = reporter.generate(SAMPLE_ALERT)

        assert isinstance(report, IncidentReport)
