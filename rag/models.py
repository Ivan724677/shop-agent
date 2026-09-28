"""Typed data contracts for retrieval, planning and evidence verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any
import uuid


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class PlanAction(StringEnum):
    ANSWER_DIRECTLY = "answer_directly"
    RETRIEVE = "retrieve"
    REWRITE_QUERY = "rewrite_query"
    VERIFY = "verify"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class PolicyDocument:
    document_id: str
    title: str
    text: str
    tags: frozenset[str]
    version: str
    effective_from: date
    effective_to: date | None = None
    priority: int = 0
    policy_family: str = ""
    exception_of: str | None = None
    status: str = "published"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def family(self) -> str:
        return self.policy_family or self.document_id.rsplit("_", 1)[0]

    def is_active(self, as_of: date) -> bool:
        return (
            self.status == "published"
            and self.effective_from <= as_of
            and (self.effective_to is None or as_of <= self.effective_to)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.document_id,
            "title": self.title,
            "text": self.text,
            "tags": sorted(self.tags),
            "version": self.version,
            "effective_from": self.effective_from.isoformat(),
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "priority": self.priority,
            "policy_family": self.family,
            "exception_of": self.exception_of,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    as_of: date = date(2026, 7, 16)
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    required_tags: frozenset[str] = frozenset()
    permission_scopes: frozenset[str] = frozenset({"policy:read"})
    pass_number: int = 1


@dataclass
class RetrievedEvidence:
    document: PolicyDocument
    vector_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    retrieval_pass: int = 1
    query: str = ""
    chunk_id: str | None = None
    source_uri: str | None = None
    source_locator: str | None = None

    @property
    def evidence_id(self) -> str:
        return self.document.document_id

    @property
    def citation_id(self) -> str:
        """Prefer chunk-level citations while preserving document compatibility."""

        return self.chunk_id or self.document.document_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "title": self.document.title,
            "version": self.document.version,
            "vector_score": round(self.vector_score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "rerank_score": round(self.rerank_score, 6),
            "matched_terms": list(self.matched_terms),
            "retrieval_pass": self.retrieval_pass,
            "query": self.query,
            "chunk_id": self.chunk_id,
            "citation_id": self.citation_id,
            "source_uri": self.source_uri,
            "source_locator": self.source_locator,
        }


@dataclass
class EvidenceValidation:
    accepted: list[RetrievedEvidence] = field(default_factory=list)
    rejected_ids: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    
    @property
    def sufficient(self) -> bool:
        return bool(self.accepted) and not self.conflicts and not self.missing_aspects

    @property
    def evidence_ids(self) -> list[str]:
        return [item.evidence_id for item in self.accepted]

    @property
    def citation_ids(self) -> list[str]:
        return [item.citation_id for item in self.accepted]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "accepted_ids": self.evidence_ids,
            "rejected_ids": list(self.rejected_ids),
            "conflicts": list(self.conflicts),
            "missing_aspects": list(self.missing_aspects),
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class PlanDecision:
    action: PlanAction
    reason: str
    confidence: float = 0.0
    query: str = ""
    retrieval_mode: str = "hybrid"
    metadata_filter: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticRAGResult:
    status: str
    answer: str
    evidence: list[RetrievedEvidence] = field(default_factory=list)
    validation: EvidenceValidation = field(default_factory=EvidenceValidation)
    retrieval_count: int = 0
    planner_calls: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    model_error: str | None = None
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stop_reason: str = ""
    latency_ms: float = 0.0
    generator_calls: int = 0
    grader_calls: int = 0
    rewrite_count: int = 0
    generated_answer: Any | None = None
    citation_validation: Any | None = None

    @property
    def evidence_ids(self) -> list[str]:
        # Only validated evidence may be cited by a caller. Raw retrieved
        # documents remain available in trace for debugging, but stale or
        # conflicting documents must not leak into the final citation set.
        return self.validation.evidence_ids

    @property
    def citation_ids(self) -> list[str]:
        return self.validation.citation_ids
