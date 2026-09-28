"""Offline comparison harness for retrieval variants and Agentic RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from rag.agent import AgenticRAG
from rag.models import PolicyDocument, RetrievalQuery
from rag.planner import HeuristicRAGPlanner
from rag.retrieval import BM25Retriever, HybridRetriever, PolicyCorpus


@dataclass(frozen=True)
class RAGEvaluationCase:
    case_id: str
    category: str
    query: str
    expected_evidence: list[str]
    metadata_filter: dict[str, Any]
    as_of: date = date(2026, 7, 16)
    policy_sensitive: bool = True
    expected_status: str = "GROUNDED"
    expected_conflict: bool = False
    required_aspects: list[str] | None = None
    expect_no_retrieval: bool = False
    extra_documents: list[dict[str, Any]] = field(default_factory=list)
    reviewed: bool = True
    review_notes: str = ""


def evaluate_case(corpus: PolicyCorpus, case: RAGEvaluationCase) -> list[dict[str, Any]]:
    case_corpus = _corpus_for_case(corpus, case)
    query = RetrievalQuery(
        case.query,
        as_of=case.as_of,
        metadata_filter=case.metadata_filter,
    )
    variants: list[tuple[str, Any]] = [
        ("bm25", BM25Retriever(case_corpus)),
        ("hybrid", HybridRetriever(case_corpus)),
    ]
    rows: list[dict[str, Any]] = []
    for name, retriever in variants:
        results = retriever.search(query, top_k=5)
        validation = _validate(case_corpus, query, results, case)
        found = {item.evidence_id for item in results}
        rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "variant": name,
            "hit": _case_hit(case, set(validation.evidence_ids), validation),
            "retrieval_hit": _expected_retrieved(case, found),
            "accepted_evidence_ids": validation.evidence_ids,
            "returned_ids": sorted(found),
            "accepted_evidence_rate": _accepted_rate(results, validation),
            "conflict_detected": bool(validation.conflicts),
            "stale_rejected": "STALE_OR_LOWER_PRECEDENCE_EVIDENCE_REJECTED" in validation.reason_codes,
            "status": "GROUNDED" if validation.sufficient else "UNCERTAIN",
            "expected_status_match": ("GROUNDED" if validation.sufficient else "UNCERTAIN") == case.expected_status,
            "expected_conflict_match": bool(validation.conflicts) == case.expected_conflict,
            "expected_no_retrieval_match": False == case.expect_no_retrieval,
            "retrieval_count": 1,
            "planner_calls": 0,
            "grader_calls": 0,
            "rewrite_count": 0,
            "validation": validation.as_dict(),
        })
    agentic = AgenticRAG(case_corpus, planner=HeuristicRAGPlanner(), as_of=case.as_of)
    result = agentic.run(
        case.query,
        metadata_filter=case.metadata_filter,
        required_aspects=case.required_aspects,
        as_of=case.as_of,
    )
    rows.append({
        "case_id": case.case_id,
        "category": case.category,
        "variant": "agentic",
        "hit": _case_hit(case, set(result.evidence_ids), result.validation),
        "retrieval_hit": _expected_retrieved(
            case,
            {item.evidence_id for item in result.evidence},
        ),
        "accepted_evidence_ids": result.evidence_ids,
        "returned_ids": result.evidence_ids,
        "accepted_evidence_rate": _accepted_rate(result.evidence, result.validation),
        "conflict_detected": bool(result.validation.conflicts),
        "stale_rejected": "STALE_OR_LOWER_PRECEDENCE_EVIDENCE_REJECTED" in result.validation.reason_codes,
        "expected_status_match": result.status == case.expected_status,
        "expected_conflict_match": bool(result.validation.conflicts) == case.expected_conflict,
        "expected_no_retrieval_match": (result.retrieval_count == 0) == case.expect_no_retrieval,
        "status": result.status,
        "retrieval_count": result.retrieval_count,
        "planner_calls": result.planner_calls,
        "grader_calls": result.grader_calls,
        "rewrite_count": result.rewrite_count,
        "rewrite_improved": any(
            item.get("event") == "rewrite_outcome" and item.get("coverage_improved")
            for item in result.trace
        ),
        "validation": result.validation.as_dict(),
    })
    return rows


def _corpus_for_case(corpus: PolicyCorpus, case: RAGEvaluationCase) -> PolicyCorpus:
    """Add only case-local documents, so a conflict fixture cannot poison other cases."""

    if not case.extra_documents:
        return corpus
    documents = list(corpus.documents)
    for raw in case.extra_documents:
        documents.append(
            PolicyDocument(
                document_id=str(raw["id"]),
                title=str(raw.get("title", raw["id"])),
                text=str(raw["text"]),
                tags=frozenset(str(tag) for tag in raw.get("tags", [])),
                version=str(raw.get("version", "unknown")),
                effective_from=date.fromisoformat(str(raw["effective_from"])),
                effective_to=(
                    date.fromisoformat(str(raw["effective_to"]))
                    if raw.get("effective_to")
                    else None
                ),
                priority=int(raw.get("priority", 0)),
                policy_family=str(raw.get("policy_family", "")),
                exception_of=raw.get("exception_of"),
                status=str(raw.get("status", "published")),
                metadata=dict(raw.get("metadata", {})),
            )
        )
    return PolicyCorpus(documents)


def _validate(
    corpus: PolicyCorpus,
    query: RetrievalQuery,
    results: list[Any],
    case: RAGEvaluationCase,
):
    from rag.evidence import EvidenceValidator

    return EvidenceValidator(corpus).validate(
        query,
        results,
        required_aspects=case.required_aspects,
    )


def _accepted_rate(results: list[Any], validation: Any) -> float:
    if not results:
        return 0.0
    return len(validation.accepted) / len(results)


def _case_hit(case: RAGEvaluationCase, found: set[str], validation: Any) -> bool:
    expected = set(case.expected_evidence)
    if expected:
        return expected.issubset(found)
    # An empty expected set means “no usable evidence”, which is important for
    # conflict and irrelevant-query fixtures; an empty subset check would
    # otherwise make every such case pass trivially.
    return not validation.accepted and (not case.expected_conflict or bool(validation.conflicts))


def _expected_retrieved(case: RAGEvaluationCase, found: set[str]) -> bool:
    """Raw recall metric, intentionally separate from validated evidence hit."""

    expected = set(case.expected_evidence)
    return expected.issubset(found) if expected else not case.expected_conflict


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({row["variant"] for row in rows})
    output: list[dict[str, Any]] = []
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        output.append({
            "variant": variant,
            "cases": len(selected),
            "hit_rate": sum(bool(row["hit"]) for row in selected) / len(selected),
            "raw_retrieval_hit_rate": sum(bool(row["retrieval_hit"]) for row in selected) / len(selected),
            "accepted_evidence_rate": sum(row["accepted_evidence_rate"] for row in selected) / len(selected),
            "conflict_detection_rate": sum(bool(row["conflict_detected"]) for row in selected) / len(selected),
            "stale_rejection_rate": sum(bool(row["stale_rejected"]) for row in selected) / len(selected),
            "low_relevance_rejection_rate": sum(
                "LOW_RELEVANCE_EVIDENCE_REJECTED" in row["validation"].get("reason_codes", [])
                for row in selected
            ) / len(selected),
            "average_retrieval_count": sum(row["retrieval_count"] for row in selected) / len(selected),
            "no_retrieval_rate": sum(row.get("retrieval_count", 0) == 0 for row in selected) / len(selected),
            "average_planner_calls": sum(row.get("planner_calls", 0) for row in selected) / len(selected),
            "average_grader_calls": sum(row.get("grader_calls", 0) for row in selected) / len(selected),
            "rewrite_rate": sum(row.get("rewrite_count", 0) > 0 for row in selected) / len(selected),
            "rewrite_success_rate": (
                sum(bool(row.get("rewrite_improved")) for row in selected if row.get("rewrite_count", 0) > 0)
                / sum(row.get("rewrite_count", 0) > 0 for row in selected)
                if any(row.get("rewrite_count", 0) > 0 for row in selected) else 0.0
            ),
            "grounded_answer_rate": sum(row.get("status") == "GROUNDED" for row in selected) / len(selected),
            "uncertainty_rate": sum(row.get("status") == "UNCERTAIN" for row in selected) / len(selected),
            "expected_status_accuracy": sum(bool(row.get("expected_status_match")) for row in selected) / len(selected),
            "expected_conflict_accuracy": sum(bool(row.get("expected_conflict_match")) for row in selected) / len(selected),
            "expected_no_retrieval_accuracy": sum(bool(row.get("expected_no_retrieval_match")) for row in selected) / len(selected),
        })
    return output


def summarize_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    pairs = sorted({(row["variant"], row.get("category", "uncategorized")) for row in rows})
    for variant, category in pairs:
        selected = [
            row for row in rows
            if row["variant"] == variant and row.get("category", "uncategorized") == category
        ]
        output.append({
            "variant": variant,
            "category": category,
            "cases": len(selected),
            "hit_rate": sum(bool(row["hit"]) for row in selected) / len(selected),
            "expected_status_accuracy": sum(bool(row.get("expected_status_match")) for row in selected) / len(selected),
            "expected_no_retrieval_accuracy": sum(bool(row.get("expected_no_retrieval_match")) for row in selected) / len(selected),
            "average_retrieval_count": sum(row.get("retrieval_count", 0) for row in selected) / len(selected),
            "rewrite_rate": sum(row.get("rewrite_count", 0) > 0 for row in selected) / len(selected),
        })
    return output


def load_cases(path) -> list[RAGEvaluationCase]:
    """Load JSON fixtures while keeping dates typed inside the evaluator."""

    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        raw = payload["cases"]
        profiles = dict(payload.get("profiles", {}))
        defaults = dict(payload.get("defaults", {}))
    else:
        raise ValueError("RAG 评测文件必须是数组，或包含 cases/profiles 的对象。")
    cases: list[RAGEvaluationCase] = []
    for source in raw:
        if not isinstance(source, dict):
            raise ValueError("RAG 评测 case 必须是对象。")
        profile_name = source.get("profile")
        profile = profiles.get(profile_name, {}) if profile_name else {}
        if profile_name and not isinstance(profile, dict):
            raise ValueError(f"RAG 评测 profile 无效：{profile_name}")
        item = {**defaults, **profile, **source}
        if not isinstance(item, dict):
            raise ValueError("RAG 评测 case 必须是对象。")
        cases.append(
            RAGEvaluationCase(
                case_id=str(item["case_id"]),
                category=str(item.get("category", "uncategorized")),
                query=str(item["query"]),
                expected_evidence=[str(value) for value in item.get("expected_evidence", [])],
                metadata_filter=dict(item.get("metadata_filter", {})),
                as_of=date.fromisoformat(str(item.get("as_of", "2026-07-16"))),
                policy_sensitive=bool(item.get("policy_sensitive", True)),
                expected_status=str(item.get("expected_status", "GROUNDED")),
                expected_conflict=bool(item.get("expected_conflict", False)),
                required_aspects=(
                    [str(value) for value in item["required_aspects"]]
                    if item.get("required_aspects") is not None
                    else None
                ),
                expect_no_retrieval=bool(item.get("expect_no_retrieval", False)),
                extra_documents=list(item.get("extra_documents", [])),
                reviewed=bool(item.get("reviewed", False)),
                review_notes=str(item.get("review_notes", "")),
            )
        )
    return cases
