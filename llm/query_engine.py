"""
Natural language query engine — "why did unit 3 fail?"

Wires together:
  1. RAGStore.retrieve()  →  relevant context chunks
  2. DeepSeek API (OpenAI-compatible)  →  grounded answer

Model: deepseek-chat  (DeepSeek-V3, fast and cost-efficient)
API is OpenAI-compatible — uses openai SDK pointed at DeepSeek's base URL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from llm.prompts import SYSTEM_QUERY_ENGINE, build_query_prompt
from llm.rag import RAGStore

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass
class QueryResult:
    answer: str
    retrieved_context: list[dict]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    model: str


class QueryEngine:
    """
    NL query engine backed by DeepSeek + ChromaDB RAG.

    Usage:
        engine = QueryEngine(rag_store=store)
        result = engine.query("why did machine M14860 trigger an alert?")
        print(result.answer)
    """

    def __init__(
        self,
        rag_store: RAGStore,
        model: str = "deepseek-chat",
        max_tokens: int = 1024,
        n_context_docs: int = 5,
        api_key: str | None = None,
    ) -> None:
        self.rag_store = rag_store
        self.model = model
        self.max_tokens = max_tokens
        self.n_context_docs = n_context_docs
        self._client = OpenAI(
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
        )

    def query(
        self,
        question: str,
        machine_id: str | None = None,
        recent_alerts: list[dict] | None = None,
    ) -> QueryResult:
        """
        Answer a natural language question about machine health.

        Parameters
        ----------
        question      : The operator's question in plain English or German.
        machine_id    : If provided, also retrieves machine-specific alert history.
        recent_alerts : Recent alert dicts to include in context (optional).
        """
        # 1. Retrieve relevant context
        context_docs = self.rag_store.retrieve(question, n_results=self.n_context_docs)

        if machine_id:
            machine_alerts = self.rag_store.retrieve_for_machine(machine_id, n_results=3)
            seen_texts = {d["text"] for d in context_docs}
            for doc in machine_alerts:
                if doc["text"] not in seen_texts:
                    context_docs.append(doc)
                    seen_texts.add(doc["text"])

        # 2. Build prompt
        user_prompt = build_query_prompt(
            user_question=question,
            retrieved_context=context_docs,
            recent_alerts=recent_alerts,
        )

        # 3. Call DeepSeek
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_QUERY_ENGINE},
                {"role": "user", "content": user_prompt},
            ],
        )

        usage = response.usage
        return QueryResult(
            answer=response.choices[0].message.content,
            retrieved_context=context_docs,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cache_read_tokens=getattr(usage, "prompt_cache_hit_tokens", 0),
            cache_creation_tokens=getattr(usage, "prompt_cache_miss_tokens", 0),
            model=response.model,
        )

    def query_stream(
        self,
        question: str,
        machine_id: str | None = None,
        recent_alerts: list[dict] | None = None,
    ):
        """
        Streaming variant — yields text chunks as they arrive.
        Useful for the FastAPI SSE endpoint.
        """
        context_docs = self.rag_store.retrieve(question, n_results=self.n_context_docs)
        user_prompt = build_query_prompt(
            user_question=question,
            retrieved_context=context_docs,
            recent_alerts=recent_alerts,
        )

        stream = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_QUERY_ENGINE},
                {"role": "user", "content": user_prompt},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
