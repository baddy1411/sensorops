"""
LLM API router — mounted into the main FastAPI app at /api/v1/llm/

Endpoints:
  POST /api/v1/llm/query           — NL question → grounded answer
  POST /api/v1/llm/query/stream    — SSE streaming variant
  POST /api/v1/llm/report          — alert dict → structured incident report
  POST /api/v1/llm/index/alerts    — index a batch of alerts into RAG
  GET  /api/v1/llm/rag/stats       — collection counts
"""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm.incident_reporter import IncidentReport, IncidentReporter
from llm.query_engine import QueryEngine, QueryResult
from llm.rag import RAGStore

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


# ---------------------------------------------------------------------------
# Dependency injection — singletons shared across requests
# ---------------------------------------------------------------------------

_rag_store: RAGStore | None = None
_query_engine: QueryEngine | None = None
_incident_reporter: IncidentReporter | None = None


def get_rag_store() -> RAGStore:
    global _rag_store
    if _rag_store is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
        _rag_store = RAGStore(persist_dir=persist_dir)
        _rag_store.index_failure_modes()
    return _rag_store


def get_query_engine(store: RAGStore = Depends(get_rag_store)) -> QueryEngine:
    global _query_engine
    if _query_engine is None:
        _query_engine = QueryEngine(
            rag_store=store,
            model=os.getenv("SENSOROPS_LLM_MODEL", "deepseek-chat"),
        )
    return _query_engine


def get_incident_reporter(store: RAGStore = Depends(get_rag_store)) -> IncidentReporter:
    global _incident_reporter
    if _incident_reporter is None:
        _incident_reporter = IncidentReporter(
            rag_store=store,
            model=os.getenv("SENSOROPS_LLM_MODEL", "deepseek-chat"),
        )
    return _incident_reporter


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        examples=["Why did machine M14860 trigger a HIGH severity alert?"],
    )
    machine_id: str | None = Field(None, examples=["M14860"])
    recent_alerts: list[dict] | None = None


class QueryResponse(BaseModel):
    answer: str
    context_sources: list[str]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    model: str


class ReportRequest(BaseModel):
    alert: dict = Field(..., description="Alert dict from the alerts asset")
    model_version: str = Field(default="unknown")


class IndexAlertsRequest(BaseModel):
    alerts: list[dict] = Field(..., min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    engine: QueryEngine = Depends(get_query_engine),
) -> QueryResponse:
    """
    Answer a natural language question about machine health.

    Retrieves relevant context from the RAG store (failure mode cards,
    maintenance manuals, alert history) and generates a grounded answer
    using Claude.
    """
    try:
        result: QueryResult = engine.query(
            question=request.question,
            machine_id=request.machine_id,
            recent_alerts=request.recent_alerts,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM query failed: {exc}",
        )

    return QueryResponse(
        answer=result.answer,
        context_sources=[d["source"] for d in result.retrieved_context],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        model=result.model,
    )


@router.post("/query/stream")
def query_stream(
    request: QueryRequest,
    engine: QueryEngine = Depends(get_query_engine),
):
    """
    Streaming variant of /query — returns Server-Sent Events.
    Each event is a text chunk from Claude as it generates.
    """
    def _generate():
        try:
            for chunk in engine.query_stream(
                question=request.question,
                machine_id=request.machine_id,
                recent_alerts=request.recent_alerts,
            ):
                yield f"data: {chunk}\n\n"
        except Exception as exc:
            yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/report", response_model=IncidentReport)
def generate_report(
    request: ReportRequest,
    reporter: IncidentReporter = Depends(get_incident_reporter),
) -> IncidentReport:
    """
    Generate a structured incident report from an alert dict.

    Returns a validated JSON report with:
    - Root cause analysis
    - Prioritised action list
    - Affected components
    - EU AI Act audit fields (model_version, dataset_hash)
    """
    try:
        report = reporter.generate(
            alert=request.alert,
            model_version=request.model_version,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Report generation failed: {exc}",
        )
    return report


@router.post("/index/alerts", status_code=status.HTTP_201_CREATED)
def index_alerts(
    request: IndexAlertsRequest,
    store: RAGStore = Depends(get_rag_store),
) -> dict:
    """Index a batch of alert dicts into the RAG store."""
    for alert in request.alerts:
        store.index_alert(alert)
    return {"indexed": len(request.alerts)}


@router.get("/rag/stats")
def rag_stats(store: RAGStore = Depends(get_rag_store)) -> dict:
    """Return document counts for each RAG collection."""
    return store.stats
