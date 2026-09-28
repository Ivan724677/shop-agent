"""Public Agentic RAG facade backed by the explicit graph executor."""

from __future__ import annotations

import threading
from datetime import date
from typing import Any

from .generation import AnswerGenerator, DeterministicAnswerGenerator
from .grading import DocumentGrader, HeuristicDocumentGrader
from .graph import AgenticRAGGraph
from .models import AgenticRAGResult, RetrievedEvidence
from .monitoring import RAGMonitor
from .planner import HeuristicRAGPlanner, RAGPlanner
from .retrieval import PolicyCorpus, Retriever


class AgenticRAGError(ValueError):
    pass


class AgenticRAG:
    """Stable stage-six API; internal execution is an auditable node graph."""

    def __init__(
        self,
        corpus: PolicyCorpus,
        planner: RAGPlanner | None = None,
        *,
        max_steps: int = 6,
        max_retrievals: int = 2,
        as_of: date = date(2026, 7, 16),
        retriever: Retriever | None = None,
        grader: DocumentGrader | None = None,
        generator: AnswerGenerator | None = None,
        monitor: RAGMonitor | None = None,
        max_planner_calls: int | None = None,
        max_generator_calls: int = 1,
        max_latency_ms: float | None = None,
    ) -> None:
        if max_steps <= 0 or max_retrievals <= 0 or max_generator_calls < 0:
            raise ValueError("Agentic RAG 的步数和检索次数必须大于 0。")
        if max_planner_calls is not None and max_planner_calls <= 0:
            raise ValueError("Agent 规划预算必须大于 0。")
        self.corpus = corpus
        self.as_of = as_of
        self._evidence_cache: dict[tuple[str, ...], list[RetrievedEvidence]] = {}
        self._cache_lock = threading.RLock()
        self.graph = AgenticRAGGraph(
            corpus,
            planner or HeuristicRAGPlanner(),
            grader or HeuristicDocumentGrader(),
            generator or DeterministicAnswerGenerator(),
            retriever=retriever,
            monitor=monitor,
            max_steps=max_steps,
            max_retrievals=max_retrievals,
            max_planner_calls=max_planner_calls if max_planner_calls is not None else max_steps,
            max_generator_calls=max_generator_calls,
            max_latency_ms=max_latency_ms,
            evidence_cache=self._evidence_cache,
            cache_lock=self._cache_lock,
        )

    def run(
        self,
        user_query: str,
        *,
        metadata_filter: dict[str, Any] | None = None,
        required_aspects: list[str] | None = None,
        permission_scopes: set[str] | frozenset[str] | None = None,
        as_of: date | None = None,
    ) -> AgenticRAGResult:
        if not user_query.strip():
            raise AgenticRAGError("RAG 查询不能为空。")
        return self.graph.run(
            user_query,
            metadata_filter=metadata_filter,
            required_aspects=required_aspects,
            permission_scopes=permission_scopes,
            as_of=as_of or self.as_of,
        )

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._evidence_cache.clear()
