"""Agentic RAG controller with multi-step retrieval and grounded stopping."""

from __future__ import annotations

import json
import contextvars
import threading
import time
import uuid
from datetime import date
from typing import Any

from .citation import CitationValidator
from .evidence import EvidenceValidator
from .generation import AnswerGenerator, DeterministicAnswerGenerator
from .monitoring import MonitorAgent, RAGMonitor, RAGTelemetry
from .models import (
    AgenticRAGResult,
    PlanAction,
    PlanDecision,
    RetrievalQuery,
    RetrievedEvidence,
)
from .planner import HeuristicRAGPlanner, RAGPlanner
from .retrieval import HybridRetriever, PolicyCorpus, Retriever, VectorRetriever
from .rewrite import QueryRewriter


class AgenticRAGError(ValueError):
    pass


_run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("rag_run_id", default="")
_started_at_var: contextvars.ContextVar[float] = contextvars.ContextVar("rag_started_at", default=0.0)
_generator_calls_var: contextvars.ContextVar[int] = contextvars.ContextVar("rag_generator_calls", default=0)
_citation_validation_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "rag_citation_validation", default=None
)


class AgenticRAG:
    """The planner owns the next action; validators own truth and safety."""

    def __init__(
        self,
        corpus: PolicyCorpus,
        planner: RAGPlanner | None = None,
        *,
        max_steps: int = 6,
        max_retrievals: int = 2,
        as_of: date = date(2026, 7, 16),
        retriever: Retriever | None = None,
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
        self.planner = planner or HeuristicRAGPlanner()
        self.vector_retriever = VectorRetriever(corpus)
        self.hybrid_retriever = HybridRetriever(corpus)
        self.retriever = retriever
        self.validator = EvidenceValidator(corpus)
        self.rewriter = QueryRewriter()
        self.max_steps = max_steps
        self.max_retrievals = max_retrievals
        self.as_of = as_of
        self.generator = generator or DeterministicAnswerGenerator()
        self.citation_validator = CitationValidator()
        self.monitor = monitor
        self.monitor_agent = MonitorAgent()
        self.max_planner_calls = max_planner_calls if max_planner_calls is not None else max_steps
        self.max_generator_calls = max_generator_calls
        self.max_latency_ms = max_latency_ms
        self._evidence_cache: dict[tuple[str, str, str, str], list[RetrievedEvidence]] = {}
        self._cache_lock = threading.RLock()

    def run(
        self,
        user_query: str,
        *,
        metadata_filter: dict[str, Any] | None = None,
        required_aspects: list[str] | None = None,
        as_of: date | None = None,
    ) -> AgenticRAGResult:
        if not user_query.strip():
            raise AgenticRAGError("RAG 查询不能为空。")
        _run_id_var.set(str(uuid.uuid4()))
        _started_at_var.set(time.perf_counter())
        _generator_calls_var.set(0)
        _citation_validation_var.set(None)
        metadata_filter = dict(metadata_filter or {})
        as_of = as_of or self.as_of
        policy_sensitive = self._policy_sensitive(user_query)
        observation: dict[str, Any] = {
            "user_query": user_query,
            "policy_sensitive": policy_sensitive,
            "metadata_filter": metadata_filter,
            "retrieval_count": 0,
            "planner_step": 0,
            "evidence": [],
            "evidence_sufficient": False,
            "has_conflict": False,
            "rewritten": False,
        }
        trace: list[dict[str, Any]] = []
        all_evidence: list[RetrievedEvidence] = []
        validation = self.validator.validate(
            RetrievalQuery(user_query, as_of=as_of, metadata_filter=metadata_filter),
            [],
            required_aspects,
        )
        cache_key = (
            user_query.replace(" ", "").lower(),
            _metadata_key(metadata_filter),
            _aspects_key(required_aspects),
            as_of.isoformat(),
        )
        with self._cache_lock:
            cached = list(self._evidence_cache.get(cache_key, []))
        if cached:
            all_evidence = list(cached)
            validation = self.validator.validate(
                RetrievalQuery(user_query, as_of=as_of, metadata_filter=metadata_filter),
                all_evidence,
                required_aspects,
            )
        model_error: str | None = None
        planner_calls = 0

        for step in range(1, self.max_steps + 1):
            if planner_calls >= self.max_planner_calls:
                return self._result(
                    "UNCERTAIN" if policy_sensitive else "NO_RETRIEVAL",
                    "已达到 Agent 规划预算，暂时无法安全形成结论。",
                    all_evidence,
                    validation,
                    trace,
                    planner_calls,
                    model_error,
                    stop_reason="PLANNER_BUDGET_EXHAUSTED",
                )
            if self.max_latency_ms is not None and (time.perf_counter() - _started_at_var.get()) * 1000 >= self.max_latency_ms:
                return self._result(
                    "UNCERTAIN" if policy_sensitive else "NO_RETRIEVAL",
                    "已达到 Agent 时间预算，暂时无法安全形成结论。",
                    all_evidence,
                    validation,
                    trace,
                    planner_calls,
                    model_error,
                    stop_reason="LATENCY_BUDGET_EXHAUSTED",
                )
            observation["planner_step"] = step
            observation["evidence"] = [item.as_dict() for item in all_evidence]
            observation["evidence_sufficient"] = validation.sufficient
            observation["has_conflict"] = bool(validation.conflicts)
            try:
                decision = self.planner.plan(self._safe_observation(observation))
                planner_calls += 1
            except Exception as exc:
                model_error = f"{type(exc).__name__}: {exc}"
                trace.append({"step": step, "event": "planner_error", "error": model_error})
                if policy_sensitive and not validation.sufficient:
                    return self._result(
                        "UNCERTAIN",
                        "暂时无法从政策知识库获得足够可靠的证据，建议转人工核实。",
                        all_evidence,
                        validation,
                        trace,
                        planner_calls,
                        model_error,
                    )
                return self._result(
                    "NO_RETRIEVAL",
                    "当前问题不需要查询政策知识库。",
                    all_evidence,
                    validation,
                    trace,
                    planner_calls,
                    model_error,
                )
            trace.append(
                {
                    "step": step,
                    "event": "plan",
                    "action": decision.action.value,
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "query": decision.query,
                    "retrieval_mode": decision.retrieval_mode,
                    "metadata_filter": dict(decision.metadata_filter),
                    "usage": dict(decision.usage),
                }
            )
            decision = self._guard_decision(decision, policy_sensitive, validation, all_evidence)
            if decision.action == PlanAction.ANSWER_DIRECTLY:
                if policy_sensitive and not validation.sufficient:
                    if observation["retrieval_count"] < self.max_retrievals:
                        decision = PlanDecision(
                            action=PlanAction.REWRITE_QUERY,
                            reason="策略阻止无证据回答，要求 Agent 继续补充检索",
                            query=user_query,
                            metadata_filter=metadata_filter,
                        )
                    else:
                        return self._result(
                            "UNCERTAIN",
                            "现有证据不足以安全判断该政策问题，建议转人工核实。",
                            all_evidence,
                            validation,
                            trace,
                            planner_calls,
                            model_error,
                        )
                else:
                    status = "GROUNDED" if validation.sufficient else "NO_RETRIEVAL"
                    answer = "当前问题不需要查询政策知识库。"
                    generated = None
                    if validation.sufficient:
                        if _generator_calls_var.get() >= self.max_generator_calls:
                            return self._result(
                                "UNCERTAIN",
                                "已达到回答生成预算，建议转人工核实。",
                                all_evidence,
                                validation,
                                trace,
                                planner_calls,
                                model_error,
                                stop_reason="GENERATOR_BUDGET_EXHAUSTED",
                            )
                        try:
                            generated = self.generator.generate(user_query, validation)
                            _generator_calls_var.set(_generator_calls_var.get() + 1)
                            _citation_validation_var.set(self.citation_validator.validate(generated, validation))
                            trace.append({
                                "step": step,
                                "event": "generate",
                                "usage": dict(generated.usage),
                                "citation_validation": _citation_validation_var.get().as_dict(),
                            })
                        except Exception as exc:
                            model_error = f"{type(exc).__name__}: {exc}"
                            trace.append({"step": step, "event": "generator_error", "error": model_error})
                            return self._result(
                                "UNCERTAIN",
                                "回答生成服务暂时不可用，建议转人工核实。",
                                all_evidence,
                                validation,
                                trace,
                                planner_calls,
                                model_error,
                                stop_reason="GENERATOR_ERROR",
                            )
                        if not _citation_validation_var.get().accepted:
                            return self._result(
                                "UNCERTAIN",
                                "生成内容未通过证据引用校验，建议转人工核实。",
                                all_evidence,
                                validation,
                                trace,
                                planner_calls,
                                model_error,
                                stop_reason="CITATION_VALIDATION_FAILED",
                            )
                        answer = generated.answer
                    return self._result(
                        status,
                        answer,
                        all_evidence,
                        validation,
                        trace,
                        planner_calls,
                        model_error,
                        stop_reason="GROUNDED" if validation.sufficient else "NO_RETRIEVAL",
                        generated_answer=generated,
                    )
            if decision.action == PlanAction.CLARIFY:
                return self._result("CLARIFY", "请补充商品类型、售后原因或订单相关信息。", all_evidence, validation, trace, planner_calls, model_error)
            if decision.action == PlanAction.ESCALATE:
                return self._result("UNCERTAIN", "政策证据存在冲突或无法确认适用范围，建议转人工核实。", all_evidence, validation, trace, planner_calls, model_error)
            if decision.action == PlanAction.VERIFY:
                validation = self.validator.validate(
                    RetrievalQuery(user_query, as_of=as_of, metadata_filter=metadata_filter),
                    all_evidence,
                    required_aspects,
                )
                trace.append({"step": step, "event": "evidence_verify", "validation": validation.as_dict()})
                if validation.sufficient:
                    continue
                if validation.conflicts:
                    return self._result("UNCERTAIN", "政策版本或适用范围存在冲突，无法安全给出确定结论。", all_evidence, validation, trace, planner_calls, model_error)
                continue
            if decision.action in {PlanAction.RETRIEVE, PlanAction.REWRITE_QUERY}:
                if observation["retrieval_count"] >= self.max_retrievals:
                    continue
                query = RetrievalQuery(
                    text=decision.query or user_query,
                    as_of=as_of,
                    metadata_filter=decision.metadata_filter or metadata_filter,
                    required_tags=frozenset(),
                    pass_number=observation["retrieval_count"] + 1,
                )
                if decision.action == PlanAction.REWRITE_QUERY:
                    query = self.rewriter.as_query(query, validation.missing_aspects)
                    observation["rewritten"] = True
                retriever = self.retriever or (
                    self.vector_retriever if decision.retrieval_mode == "vector" else self.hybrid_retriever
                )
                retrieved = retriever.search(query, top_k=5)
                all_evidence = self._merge_evidence(all_evidence, retrieved)
                observation["retrieval_count"] += 1
                with self._cache_lock:
                    self._evidence_cache[cache_key] = list(all_evidence)
                validation = self.validator.validate(query, all_evidence, required_aspects)
                trace.append(
                    {
                        "step": step,
                        "event": "retrieve",
                        "pass": observation["retrieval_count"],
                        "query": query.text,
                        "mode": decision.retrieval_mode,
                        "results": [item.as_dict() for item in retrieved],
                        "validation": validation.as_dict(),
                    }
                )
                continue
        return self._result(
            "UNCERTAIN" if policy_sensitive else "NO_RETRIEVAL",
            "检索步数已达到上限，暂时无法安全形成政策结论。" if policy_sensitive else "当前问题不需要查询政策知识库。",
            all_evidence,
            validation,
            trace,
            planner_calls,
            model_error,
        )

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._evidence_cache.clear()

    @staticmethod
    def _policy_sensitive(query: str) -> bool:
        normalized = query.replace(" ", "")
        return any(word in normalized for word in HeuristicRAGPlanner.POLICY_WORDS)

    @staticmethod
    def _safe_observation(observation: dict[str, Any]) -> dict[str, Any]:
        # Planner sees evidence metadata and validation reasons, not raw hidden
        # application secrets or arbitrary internal records.
        return {
            "user_query": observation["user_query"],
            "policy_sensitive": observation["policy_sensitive"],
            "metadata_filter": observation["metadata_filter"],
            "retrieval_count": observation["retrieval_count"],
            "planner_step": observation["planner_step"],
            "evidence": observation["evidence"],
            "evidence_sufficient": observation["evidence_sufficient"],
            "has_conflict": observation["has_conflict"],
            "rewritten": observation["rewritten"],
        }

    def _guard_decision(self, decision: PlanDecision, policy_sensitive: bool, validation: Any, evidence: list[RetrievedEvidence]) -> PlanDecision:
        if policy_sensitive and decision.action == PlanAction.ANSWER_DIRECTLY and not validation.sufficient:
            return PlanDecision(
                action=PlanAction.REWRITE_QUERY if evidence else PlanAction.RETRIEVE,
                reason="安全护栏拒绝无证据政策回答",
                confidence=1.0,
                query=decision.query,
                retrieval_mode=decision.retrieval_mode,
                metadata_filter=decision.metadata_filter,
            )
        if decision.action == PlanAction.RETRIEVE and not decision.query:
            return PlanDecision(
                action=PlanAction.RETRIEVE,
                reason=decision.reason,
                confidence=decision.confidence,
                query="",
                retrieval_mode=decision.retrieval_mode,
                metadata_filter=decision.metadata_filter,
            )
        return decision

    @staticmethod
    def _merge_evidence(existing: list[RetrievedEvidence], incoming: list[RetrievedEvidence]) -> list[RetrievedEvidence]:
        # A persistent dense index returns chunk-level evidence.  Document IDs
        # are intentionally not the deduplication key: two useful chunks from
        # one policy document must remain independently citable.
        by_id = {item.citation_id: item for item in existing}
        for item in incoming:
            previous = by_id.get(item.citation_id)
            if previous is None or item.rerank_score > previous.rerank_score or item.vector_score > previous.vector_score:
                by_id[item.citation_id] = item
        return list(by_id.values())

    @staticmethod
    def _grounded_answer(query: str, validation: Any) -> str:
        evidence = "、".join(validation.evidence_ids)
        excerpts = "；".join(
            f"{item.document.title}：{item.document.text}"
            for item in validation.accepted
        )
        return (
            f"根据已校验的政策证据（{evidence}），当前适用规则是：{excerpts}"
            "具体资格仍需结合订单状态和商品信息确认。"
        )

    def _result(
        self,
        status,
        answer,
        evidence,
        validation,
        trace,
        planner_calls,
        model_error,
        *,
        stop_reason: str = "",
        generated_answer: Any | None = None,
    ):
        result = AgenticRAGResult(
            status=status,
            answer=answer,
            evidence=list(evidence),
            validation=validation,
            retrieval_count=sum(1 for item in trace if item.get("event") == "retrieve"),
            planner_calls=planner_calls,
            trace=trace,
            model_error=model_error,
            run_id=_run_id_var.get(),
            stop_reason=stop_reason or status,
            latency_ms=(time.perf_counter() - _started_at_var.get()) * 1000 if _started_at_var.get() else 0.0,
            generator_calls=_generator_calls_var.get(),
            generated_answer=generated_answer,
            citation_validation=_citation_validation_var.get(),
        )
        if self.monitor:
            telemetry = RAGTelemetry(
                run_id=result.run_id,
                status=result.status,
                stop_reason=result.stop_reason,
                retrieval_count=result.retrieval_count,
                planner_calls=result.planner_calls,
                generator_calls=result.generator_calls,
                latency_ms=result.latency_ms,
                evidence_count=len(result.evidence),
                accepted_evidence_count=len(result.validation.accepted),
                citation_valid=(
                    result.citation_validation.accepted
                    if result.citation_validation is not None else None
                ),
                model_error=result.model_error,
                evidence_ids=result.evidence_ids,
                citation_ids=result.citation_ids,
            )
            telemetry.alerts = self.monitor_agent.analyze(telemetry)
            self.monitor.record(telemetry)
        return result


def _metadata_key(metadata: dict[str, Any]) -> str:
    return "|".join(f"{key}={metadata[key]}" for key in sorted(metadata))


def _aspects_key(aspects: list[str] | None) -> str:
    return "|".join(sorted(set(aspects or [])))
