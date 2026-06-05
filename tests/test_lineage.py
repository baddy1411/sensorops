"""
Tests for OpenLineage emitter and EU AI Act audit logger.
All tests are offline — no OL server required.
"""

from __future__ import annotations

import json
from pathlib import Path

from lineage.audit_log import AuditLogger
from lineage.emitter import LineageEmitter

# ---------------------------------------------------------------------------
# LineageEmitter tests
# ---------------------------------------------------------------------------


class TestLineageEmitter:
    def test_emits_to_jsonl_when_no_server(self, tmp_path):
        log = tmp_path / "lineage.jsonl"
        emitter = LineageEmitter(url="", fallback_log=str(log))
        run_id = emitter.emit_start("test_job", inputs=["input.csv"])
        emitter.emit_complete(run_id, "test_job", inputs=["input.csv"], outputs=["output.parquet"])
        lines = log.read_text().splitlines()
        assert len(lines) == 2

    def test_start_event_type(self, tmp_path):
        log = tmp_path / "lineage.jsonl"
        emitter = LineageEmitter(url="", fallback_log=str(log))
        emitter.emit_start("my_job")
        record = json.loads(log.read_text().strip())
        assert record["eventType"] == "START"

    def test_complete_event_type(self, tmp_path):
        log = tmp_path / "lineage.jsonl"
        emitter = LineageEmitter(url="", fallback_log=str(log))
        run_id = emitter.emit_start("my_job")
        emitter.emit_complete(run_id, "my_job", inputs=[], outputs=["out.parquet"])
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert records[-1]["eventType"] == "COMPLETE"

    def test_fail_event_type(self, tmp_path):
        log = tmp_path / "lineage.jsonl"
        emitter = LineageEmitter(url="", fallback_log=str(log))
        emitter.emit_fail("run-123", "my_job", error="something broke")
        record = json.loads(log.read_text().strip())
        assert record["eventType"] == "FAIL"

    def test_run_id_consistent_across_events(self, tmp_path):
        log = tmp_path / "lineage.jsonl"
        emitter = LineageEmitter(url="", fallback_log=str(log))
        run_id = emitter.emit_start("my_job")
        emitter.emit_complete(run_id, "my_job", [], [])
        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert records[0]["run"]["runId"] == records[1]["run"]["runId"] == run_id

    def test_track_ingest_writes_file_hash(self, tmp_path):
        log = tmp_path / "lineage.jsonl"
        emitter = LineageEmitter(url="", fallback_log=str(log))
        run_id = emitter.emit_start("ingest")
        emitter.track_ingest(run_id, csv_path="data/raw/ai4i2020.csv", row_count=500)
        records = [json.loads(line) for line in log.read_text().splitlines()]
        complete = records[-1]
        assert complete["run"]["facets"].get("fileHash") is not None

    def test_track_scoring_has_eu_ai_act_facet(self, tmp_path):
        log = tmp_path / "lineage.jsonl"
        emitter = LineageEmitter(url="", fallback_log=str(log))
        run_id = emitter.emit_start("scoring")
        emitter.track_scoring(
            run_id=run_id,
            model_name="isolation_forest",
            model_version="v1",
            dataset_hash="abc123",
            n_samples=1000,
            n_flagged=35,
            mlflow_run_id="run-xyz",
        )
        records = [json.loads(line) for line in log.read_text().splitlines()]
        complete_facets = records[-1]["run"]["facets"]
        assert "euAiActAudit" in complete_facets
        audit = complete_facets["euAiActAudit"]
        assert audit["modelName"] == "isolation_forest"
        assert audit["datasetHash"] == "abc123"
        assert audit["nFlagged"] == 35

    def test_schema_facets_attached_to_known_datasets(self, tmp_path):
        log = tmp_path / "lineage.jsonl"
        emitter = LineageEmitter(url="", fallback_log=str(log))
        run_id = emitter.emit_start("features")
        emitter.emit_complete(
            run_id,
            "features",
            inputs=["data/processed/raw_events.parquet"],
            outputs=["data/processed/features.parquet"],
        )
        records = [json.loads(line) for line in log.read_text().splitlines()]
        complete = records[-1]
        # raw_events input should have schema facet
        input_facets = complete["inputs"][0].get("facets", {})
        assert "schema" in input_facets


# ---------------------------------------------------------------------------
# AuditLogger tests
# ---------------------------------------------------------------------------


class TestAuditLogger:
    def test_log_prediction_creates_file(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_path)
        logger.log_prediction("M1", anomaly_score=0.75, is_anomaly=True)
        assert Path(log_path).exists()

    def test_sequence_numbers_monotonic(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_path)
        for i in range(5):
            logger.log_prediction(f"M{i}", anomaly_score=0.5, is_anomaly=False)
        records = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
        seqs = [r["seq"] for r in records]
        assert seqs == list(range(1, 6))

    def test_hash_chain_valid_on_fresh_log(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_path)
        logger.log_prediction("M1", 0.8, True)
        logger.log_alert("a1", "M1", "HIGH", "HDF", 0.8)
        logger.log_pipeline_run("ingest", "run-1", "completed", row_count=500)
        valid, errors = logger.verify_chain()
        assert valid, f"Chain errors: {errors}"

    def test_tampered_record_detected(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_path)
        logger.log_prediction("M1", 0.8, True)
        logger.log_prediction("M2", 0.3, False)

        # Tamper with first record
        lines = Path(log_path).read_text().splitlines()
        rec = json.loads(lines[0])
        rec["payload"]["anomaly_score"] = 0.01  # tampered!
        lines[0] = json.dumps(rec)
        Path(log_path).write_text("\n".join(lines) + "\n")

        valid, errors = logger.verify_chain()
        assert not valid
        assert len(errors) > 0

    def test_log_promotion_has_requires_human_approval(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_path)
        logger.log_promotion(
            model_name="isolation_forest",
            from_stage="Staging",
            to_stage="Production",
            model_version="v2",
            triggered_by="operator_42",
            metrics={"f1": 0.72, "roc_auc": 0.88},
        )
        record = json.loads(Path(log_path).read_text().strip())
        assert record["payload"]["requires_human_approval"] is True

    def test_get_stats_counts_by_type(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        logger = AuditLogger(log_path=log_path)
        logger.log_prediction("M1", 0.5, False)
        logger.log_prediction("M2", 0.9, True)
        logger.log_alert("a1", "M1", "HIGH", "HDF", 0.9)
        stats = logger.get_stats()
        assert stats["PREDICTION"] == 2
        assert stats["ALERT"] == 1

    def test_seq_resumes_after_reload(self, tmp_path):
        log_path = str(tmp_path / "audit.jsonl")
        logger1 = AuditLogger(log_path=log_path)
        logger1.log_prediction("M1", 0.5, False)
        logger1.log_prediction("M2", 0.6, False)

        # Reload
        logger2 = AuditLogger(log_path=log_path)
        logger2.log_prediction("M3", 0.7, True)

        records = [json.loads(line) for line in Path(log_path).read_text().splitlines()]
        assert records[-1]["seq"] == 3
