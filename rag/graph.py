"""Explicit, typed Agentic RAG graph with auditable node boundaries."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from threading import RLock
from typing import Any

from .business_filter import BusinessFilterResult, DeterministicBusinessFilter
from .citation import CitationValidation, CitationValidator
from .evidence import EvidenceValidator
from .generation import AnswerGenerator, GeneratedAnswer
from .grading import DocumentGrader, DocumentGradingResult, infer_query_aspects
from .models import (
    AgenticRAGResult,
    EvidenceValidation,
    PlanAction,
    PlanDecision,
    RetrievalQuery,
    RetrievedEvidence,
)
from .monitoring import MonitorAgent, RAGMonitor, RAGTelemetry
from .planner import HeuristicRAGPlanner, RAGPlanner
from .retrieval import BM25Retriever, HybridRetriever, PolicyCorpus, Retriever
from .rewrite import QueryRewriter


@dataclass
class RAGGraphState:
    run_id: str
    user_query: str
    as_of: date
    metadata_filter: dict[str, Any]
    permission_scopes: frozenset[str]
    required_aspects: list[str]
    policy_sensitive: bool = False
    guardrail_missing_fields: list[str] = field(default_factory=list)
    current_query: str = ""
    retrieval_mode: str = "hybrid"
    retrieval_count: int = 0
    planner_calls: int = 0
    grader_calls: int = 0
    generator_calls: int = 0
    rewrite_count: int = 0
    rewritten: bool = False
    all_evidence: list[RetrievedEvidence] = field(default_factory=list)
    current_candidates: list[RetrievedEvidence] = field(default_factory=list)
    business_filter: BusinessFilterResult = field(default_factory=BusinessFilterResult)
    grading: DocumentGradingResult = field(default_factory=DocumentGradingResult)
    validation: EvidenceValidation = field(default_factory=EvidenceValidation)
    generated_answer: GeneratedAnswer | None = None
    citation_validation: CitationValidation | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    rewrite_history: list[dict[str, Any]] = field(default_factory=list)
    pending_rewrite: int | None = None
    model_error: str | None = None
    status: str = ""
    answer: str = ""
    stop_reason: str = ""

    def retrieval_query(self, *, text: str | None = None, pass_number: int | None = None) -> RetrievalQuery:
        return RetrievalQuery(
            text=text or self.current_query or self.user_query,
            as_of=self.as_of,
            metadata_filter=dict(self.metadata_filter),
            permission_scopes=self.permission_scopes,
            pass_number=pass_number or max(self.retrieval_count, 1),
        )


class GuardrailNode:
    name = "guardrail"

    def run(self, state: RAGGraphState) -> None:
        normalized = state.user_query.replace(" ", "")
        state.policy_sensitive = bool(state.metadata_filter or state.required_aspects) or any(
            word in normalized for word in HeuristicRAGPlanner.POLICY_WORDS
        )
        missing: list[str] = []
        if state.policy_sensitive:
            if not state.metadata_filter.get("product_type") and not any(
                word in normalized for word in ("普通", "定制")
            ):
                missing.append("product_type")
            if not state.metadata_filter.get("reason") and not any(
                word in normalized for word in ("质量", "坏", "故障", "瑕疵", "无理由", "不想要", "不合适")
            ):
                missing.append("reason")
        state.guardrail_missing_fields = missing
        state.trace.append({
            "event": "guardrail",
            "node": self.name,
            "policy_sensitive": state.policy_sensitive,
            "missing_fields": missing,
        })


class PlannerNode:
    name = "planner"

    def __init__(self, planner: RAGPlanner) -> None:
        self.planner = planner

    def run(self, state: RAGGraphState) -> PlanDecision:
        observation = {
            "user_query": state.user_query,
            "policy_sensitive": state.policy_sensitive,
            "metadata_filter": dict(state.metadata_filter),
            "retrieval_count": state.retrieval_count,
            "planner_step": state.planner_calls + 1,
            "evidence": [item.as_dict() for item in state.all_evidence],
            "evidence_sufficient": state.validation.sufficient,
            "has_conflict": bool(state.validation.conflicts or state.business_filter.conflicts),
            "rewritten": state.rewritten,
            "guardrail_missing_fields": list(state.guardrail_missing_fields),
            "semantic_missing_aspects": state.grading.missing_aspects,
        }
        decision = self.planner.plan(observation)
        state.planner_calls += 1
        state.trace.append({
            "event": "plan",
            "node": self.name,
            "step": state.planner_calls,
            "action": decision.action.value,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "query": decision.query,
            "retrieval_mode": decision.retrieval_mode,
            "metadata_filter": dict(decision.metadata_filter),
            "usage": dict(decision.usage),
        })
        return decision


class RetrieveNode:
    name = "retrieve"

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        hybrid_retriever: HybridRetriever,
        business_filter: DeterministicBusinessFilter,
        retriever: Retriever | None = None,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.hybrid_retriever = hybrid_retriever
        self.business_filter = business_filter
        self.retriever = retriever

    def run(self, state: RAGGraphState) -> None:
        state.retrieval_count += 1
        query = state.retrieval_query(pass_number=state.retrieval_count)
        retriever = self.retriever or (
            self.bm25_retriever if state.retrieval_mode in {"vector", "bm25"}
            else self.hybrid_retriever
        )
        raw = retriever.search(query, top_k=5)
        state.business_filter = self.business_filter.apply(query, raw)
        state.current_candidates = list(state.business_filter.accepted)
        state.trace.append({
            "event": "retrieve",
            "node": self.name,
            "pass": state.retrieval_count,
            "query": query.text,
            "mode": state.retrieval_mode,
            "results": [item.as_dict() for item in raw],
        })
        state.trace.append({
            "event": "business_filter",
            "node": "business_filter",
            "pass": state.retrieval_count,
            "result": state.business_filter.as_dict(),
        })


class GradeNode:
    name = "grade"

    def __init__(self, grader: DocumentGrader) -> None:
        self.grader = grader

    def run(self, state: RAGGraphState) -> None:
        query = state.retrieval_query(pass_number=state.retrieval_count)
        state.grading = self.grader.grade(query, state.current_candidates, state.required_aspects)
        state.grader_calls += 1
        relevant = state.grading.relevant_ids
        state.current_candidates = [
            item for item in state.current_candidates if item.citation_id in relevant
        ]
        state.all_evidence = merge_evidence(state.all_evidence, state.current_candidates)
        state.trace.append({
            "event": "document_grade",
            "node": self.name,
            "pass": state.retrieval_count,
            "result": state.grading.as_dict(),
        })


class RewriteNode:
    name = "rewrite"

    def __init__(self, rewriter: QueryRewriter) -> None:
        self.rewriter = rewriter

    def run(self, state: RAGGraphState, reason: str) -> bool:
        missing = list(dict.fromkeys(
            state.validation.missing_aspects
            + state.grading.missing_aspects
            + state.guardrail_missing_fields
        ))
        query = state.retrieval_query(text=state.user_query, pass_number=state.retrieval_count + 1)
        candidates = self.rewriter.rewrite(query, missing)
        if not candidates:
            state.trace.append({
                "event": "rewrite",
                "node": self.name,
                "changed": False,
                "reason": reason,
                "missing_aspects": missing,
            })
            return False
        candidate = candidates[0]
        before_ids = set(state.validation.citation_ids)
        record = {
            "attempt": state.rewrite_count + 1,
            "original_query": state.current_query or state.user_query,
            "rewritten_query": candidate.text,
            "planner_reason": reason,
            "rewrite_reason": candidate.reason,
            "missing_aspects_before": missing,
            "added_terms": candidate.added_terms,
            "evidence_before": sorted(before_ids),
            "new_evidence": [],
            "missing_aspects_after": [],
            "coverage_improved": False,
        }
        state.rewrite_history.append(record)
        state.pending_rewrite = len(state.rewrite_history) - 1
        state.current_query = candidate.text
        state.rewrite_count += 1
        state.rewritten = True
        state.trace.append({
            "event": "rewrite",
            "node": self.name,
            "changed": True,
            **record,
        })
        return True


class VerifyNode:
    name = "verify"

    def __init__(self, validator: EvidenceValidator) -> None:
        self.validator = validator

    def run(self, state: RAGGraphState) -> None:
        # Verify against the original user question and deterministic metadata,
        # not against terms injected by query rewriting.
        query = RetrievalQuery(
            state.user_query,
            as_of=state.as_of,
            metadata_filter=dict(state.metadata_filter),
            permission_scopes=state.permission_scopes,
        )
        state.validation = self.validator.validate(
            query,
            state.all_evidence,
            state.required_aspects,
        )
        state.trace.append({
            "event": "evidence_verify",
            "node": self.name,
            "validation": state.validation.as_dict(),
        })
        if state.pending_rewrite is not None:
            record = state.rewrite_history[state.pending_rewrite]
            after_ids = set(state.validation.citation_ids)
            before_missing = set(record["missing_aspects_before"])
            after_missing = set(state.validation.missing_aspects)
            record["new_evidence"] = sorted(after_ids - set(record["evidence_before"]))
            record["missing_aspects_after"] = sorted(after_missing)
            record["coverage_improved"] = (
                state.validation.sufficient
                or len(after_missing) < len(before_missing)
                or bool(record["new_evidence"])
            )
            state.trace.append({
                "event": "rewrite_outcome",
                "node": self.name,
                **record,
            })
            state.pending_rewrite = None


class GenerateNode:
    name = "generate"

    def __init__(self, generator: AnswerGenerator) -> None:
        self.generator = generator

    def run(self, state: RAGGraphState) -> None:
        state.generated_answer = self.generator.generate(state.user_query, state.validation)
        state.generator_calls += 1
        state.trace.append({
            "event": "generate",
            "node": self.name,
            "usage": dict(state.generated_answer.usage),
        })


class CitationNode:
    name = "citation"

    def __init__(self, validator: CitationValidator) -> None:
        self.validator = validator

    def run(self, state: RAGGraphState) -> None:
        if state.generated_answer is None:
            raise ValueError("Citation Node 缺少生成结果。")
        state.citation_validation = self.validator.validate(
            state.generated_answer,
            state.validation,
        )
        state.trace.append({
            "event": "citation_validate",
            "node": self.name,
            "result": state.citation_validation.as_dict(),
        })


class EscalateNode:
    name = "escalate"

    def run(self, state: RAGGraphState, reason: str, answer: str | None = None) -> None:
        state.status = "UNCERTAIN"
        state.stop_reason = reason
        state.answer = answer or "政策证据不足、冲突或模型链路异常，建议转人工核实。"
        state.trace.append({
            "event": "escalate",
            "node": self.name,
            "reason": reason,
        })


class AgenticRAGGraph:
    """Graph executor. Nodes own work; this class only owns transitions."""

    graph_definition = {
        "guardrail": ["planner"],
        "planner": ["retrieve", "rewrite", "verify", "generate", "escalate", "end"],
        "retrieve": ["business_filter"],
        "business_filter": ["grade", "escalate"],
        "grade": ["verify"],
        "rewrite": ["retrieve", "escalate"],
        "verify": ["planner", "escalate"],
        "generate": ["citation"],
        "citation": ["end", "escalate"],
        "escalate": ["end"],
    }

    def __init__(
        self,
        corpus: PolicyCorpus,
        planner: RAGPlanner,
        grader: DocumentGrader,
        generator: AnswerGenerator,
        *,
        retriever: Retriever | None = None,
        monitor: RAGMonitor | None = None,
        max_steps: int = 6,
        max_retrievals: int = 2,
        max_planner_calls: int = 6,
        max_generator_calls: int = 1,
        max_latency_ms: float | None = None,
        evidence_cache: dict[tuple[str, ...], list[RetrievedEvidence]] | None = None,
        cache_lock: RLock | None = None,
    ) -> None:
        self.corpus = corpus
        self.monitor = monitor
        self.monitor_agent = MonitorAgent()
        self.max_steps = max_steps
        self.max_retrievals = max_retrievals
        self.max_planner_calls = max_planner_calls
        self.max_generator_calls = max_generator_calls
        self.max_latency_ms = max_latency_ms
        self.cache = evidence_cache if evidence_cache is not None else {}
        self.cache_lock = cache_lock or RLock()
        business_filter = DeterministicBusinessFilter(corpus)
        self.guardrail_node = GuardrailNode()
        self.planner_node = PlannerNode(planner)
        self.retrieve_node = RetrieveNode(
            BM25Retriever(corpus),
            HybridRetriever(corpus),
            business_filter,
            retriever,
        )
        self.grade_node = GradeNode(grader)
        self.rewrite_node = RewriteNode(QueryRewriter())
        self.verify_node = VerifyNode(EvidenceValidator(corpus))
        self.generate_node = GenerateNode(generator)
        self.citation_node = CitationNode(CitationValidator())
        self.escalate_node = EscalateNode()

    def run(
        self,
        user_query: str,
        *,
        metadata_filter: dict[str, Any] | None = None,
        required_aspects: list[str] | None = None,
        permission_scopes: set[str] | frozenset[str] | None = None,
        as_of: date,
    ) -> AgenticRAGResult:
        started_at = time.perf_counter()
        query = RetrievalQuery(
            user_query,
            as_of=as_of,
            metadata_filter=dict(metadata_filter or {}),
            permission_scopes=frozenset(permission_scopes or {"policy:read"}),
        )
        state = RAGGraphState(
            run_id=str(uuid.uuid4()),
            user_query=user_query,
            current_query=user_query,
            as_of=as_of,
            metadata_filter=dict(metadata_filter or {}),
            permission_scopes=query.permission_scopes,
            required_aspects=list(required_aspects or infer_query_aspects(query)),
        )
        self.guardrail_node.run(state)
        self.verify_node.run(state)
        cache_key = self._cache_key(state)
        with self.cache_lock:
            cached = list(self.cache.get(cache_key, []))
        if cached:
            state.all_evidence = cached
            self.verify_node.run(state)

        iterations = 0
        while iterations < self.max_steps:
            iterations += 1
            if state.planner_calls >= self.max_planner_calls:
                self.escalate_node.run(state, "PLANNER_BUDGET_EXHAUSTED")
                break
            if self._latency_exhausted(started_at):
                self.escalate_node.run(state, "LATENCY_BUDGET_EXHAUSTED")
                break
            try:
                decision = self.planner_node.run(state)
            except Exception as exc:
                state.model_error = f"{type(exc).__name__}: {exc}"
                self.escalate_node.run(state, "PLANNER_ERROR")
                break
            decision = self._guard_decision(state, decision)

            if decision.action == PlanAction.ANSWER_DIRECTLY:
                if not state.policy_sensitive and not state.validation.sufficient:
                    state.status = "NO_RETRIEVAL"
                    state.answer = "当前问题不需要查询政策知识库。"
                    state.stop_reason = "NO_RETRIEVAL"
                    break
                if not state.validation.sufficient:
                    self.escalate_node.run(state, "INSUFFICIENT_EVIDENCE")
                    break
                if state.generator_calls >= self.max_generator_calls:
                    self.escalate_node.run(state, "GENERATOR_BUDGET_EXHAUSTED")
                    break
                try:
                    self.generate_node.run(state)
                    self.citation_node.run(state)
                except Exception as exc:
                    state.model_error = f"{type(exc).__name__}: {exc}"
                    self.escalate_node.run(state, "GENERATOR_OR_CITATION_ERROR")
                    break
                if not state.citation_validation or not state.citation_validation.accepted:
                    self.escalate_node.run(state, "CITATION_VALIDATION_FAILED")
                    break
                state.status = "GROUNDED"
                state.answer = state.generated_answer.answer if state.generated_answer else ""
                state.stop_reason = "GROUNDED"
                break

            if decision.action == PlanAction.CLARIFY:
                state.status = "CLARIFY"
                state.answer = "请补充商品类型、售后原因或订单相关信息。"
                state.stop_reason = "MISSING_REQUIRED_INPUT"
                break
            if decision.action == PlanAction.ESCALATE:
                self.escalate_node.run(state, "PLANNER_ESCALATED")
                break
            if decision.action == PlanAction.VERIFY:
                self.verify_node.run(state)
                if state.validation.conflicts:
                    self.escalate_node.run(state, "POLICY_VERSION_CONFLICT")
                    break
                continue
            if decision.action == PlanAction.REWRITE_QUERY:
                if state.retrieval_count >= self.max_retrievals:
                    self.escalate_node.run(state, "RETRIEVAL_BUDGET_EXHAUSTED")
                    break
                if not self.rewrite_node.run(state, decision.reason):
                    self.escalate_node.run(state, "QUERY_REWRITE_NO_CHANGE")
                    break
                state.retrieval_mode = decision.retrieval_mode
                if decision.metadata_filter:
                    state.metadata_filter.update(decision.metadata_filter)
                if not self._retrieve_grade_verify(state):
                    break
                continue
            if decision.action == PlanAction.RETRIEVE:
                if state.retrieval_count >= self.max_retrievals:
                    self.escalate_node.run(state, "RETRIEVAL_BUDGET_EXHAUSTED")
                    break
                state.current_query = decision.query or state.current_query
                state.retrieval_mode = decision.retrieval_mode
                if decision.metadata_filter:
                    state.metadata_filter.update(decision.metadata_filter)
                if not self._retrieve_grade_verify(state):
                    break
                continue

        if not state.status:
            self.escalate_node.run(state, "GRAPH_STEP_BUDGET_EXHAUSTED")
        result = self._result(state, started_at)
        self._record_monitor(result)
        return result

    def _retrieve_grade_verify(self, state: RAGGraphState) -> bool:
        try:
            self.retrieve_node.run(state)
            if state.business_filter.conflicts:
                # Continue to Verify so conflicts are recorded by the final
                # authoritative validator as well.
                self.verify_node.run(state)
                self.escalate_node.run(state, "POLICY_VERSION_CONFLICT")
                return False
            self.grade_node.run(state)
            self.verify_node.run(state)
        except Exception as exc:
            state.model_error = f"{type(exc).__name__}: {exc}"
            self.escalate_node.run(state, "RETRIEVAL_OR_GRADER_ERROR")
            return False
        with self.cache_lock:
            self.cache[self._cache_key(state)] = list(state.all_evidence)
        return True

    @staticmethod
    def _guard_decision(state: RAGGraphState, decision: PlanDecision) -> PlanDecision:
        if state.policy_sensitive and decision.action == PlanAction.ANSWER_DIRECTLY and not state.validation.sufficient:
            guarded = PlanDecision(
                action=PlanAction.REWRITE_QUERY if state.all_evidence or state.retrieval_count else PlanAction.RETRIEVE,
                reason="安全护栏拒绝无证据政策回答",
                confidence=1.0,
                query=decision.query or state.user_query,
                retrieval_mode=decision.retrieval_mode,
                metadata_filter=decision.metadata_filter,
            )
            state.trace.append({
                "event": "decision_guard",
                "node": "guardrail",
                "original_action": decision.action.value,
                "guarded_action": guarded.action.value,
                "reason": guarded.reason,
            })
            return guarded
        return decision

    def _latency_exhausted(self, started_at: float) -> bool:
        return self.max_latency_ms is not None and (
            time.perf_counter() - started_at
        ) * 1000 >= self.max_latency_ms

    @staticmethod
    def _cache_key(state: RAGGraphState) -> tuple[str, ...]:
        return (
            state.user_query.replace(" ", "").lower(),
            "|".join(f"{key}={state.metadata_filter[key]}" for key in sorted(state.metadata_filter)),
            "|".join(sorted(set(state.required_aspects))),
            "|".join(sorted(state.permission_scopes)),
            state.as_of.isoformat(),
        )

    @staticmethod
    def _result(state: RAGGraphState, started_at: float) -> AgenticRAGResult:
        return AgenticRAGResult(
            status=state.status,
            answer=state.answer,
            evidence=list(state.all_evidence),
            validation=state.validation,
            retrieval_count=state.retrieval_count,
            planner_calls=state.planner_calls,
            trace=list(state.trace),
            model_error=state.model_error,
            run_id=state.run_id,
            stop_reason=state.stop_reason,
            latency_ms=(time.perf_counter() - started_at) * 1000,
            generator_calls=state.generator_calls,
            grader_calls=state.grader_calls,
            rewrite_count=state.rewrite_count,
            generated_answer=state.generated_answer,
            citation_validation=state.citation_validation,
        )

    def _record_monitor(self, result: AgenticRAGResult) -> None:
        if not self.monitor:
            return
        telemetry = RAGTelemetry(
            run_id=result.run_id,
            status=result.status,
            stop_reason=result.stop_reason,
            retrieval_count=result.retrieval_count,
            planner_calls=result.planner_calls,
            generator_calls=result.generator_calls,
            grader_calls=result.grader_calls,
            rewrite_count=result.rewrite_count,
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


def merge_evidence(
    existing: list[RetrievedEvidence],
    incoming: list[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    by_id = {item.citation_id: item for item in existing}
    for item in incoming:
        previous = by_id.get(item.citation_id)
        if previous is None or (
            item.rerank_score,
            item.vector_score,
            item.lexical_score,
        ) > (
            previous.rerank_score,
            previous.vector_score,
            previous.lexical_score,
        ):
            by_id[item.citation_id] = item
    return list(by_id.values())
