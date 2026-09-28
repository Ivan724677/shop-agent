"""Deterministic policy applicability gate before semantic LLM grading."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .models import PolicyDocument, RetrievalQuery, RetrievedEvidence
from .retrieval import PolicyCorpus


@dataclass(frozen=True)
class BusinessFilterResult:
    accepted: list[RetrievedEvidence] = field(default_factory=list)
    rejected: dict[str, list[str]] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_citation_ids": [item.citation_id for item in self.accepted],
            "rejected": {key: list(value) for key, value in self.rejected.items()},
            "conflicts": list(self.conflicts),
        }


class DeterministicBusinessFilter:
    """Owns business truth; an LLM grader cannot override this result.

    Checks publication status, effective dates, product/reason applicability,
    read permission and same-precedence version conflicts. EvidenceValidator
    repeats the authoritative precedence checks after semantic grading.
    """

    def __init__(self, corpus: PolicyCorpus) -> None:
        self.corpus = corpus

    def apply(
        self,
        query: RetrievalQuery,
        evidence: list[RetrievedEvidence],
    ) -> BusinessFilterResult:
        conflicts = self.detect_conflicts(query)
        conflict_families = {
            conflict.split(":", 2)[1]
            for conflict in conflicts
            if conflict.count(":") >= 2
        }
        accepted: list[RetrievedEvidence] = []
        rejected: dict[str, list[str]] = {}
        for item in evidence:
            reasons = self._rejection_reasons(query, item.document, conflict_families)
            if reasons:
                rejected[item.citation_id] = reasons
            else:
                accepted.append(item)
        return BusinessFilterResult(accepted, rejected, conflicts)

    def detect_conflicts(self, query: RetrievalQuery) -> list[str]:
        by_family: dict[str, list[PolicyDocument]] = defaultdict(list)
        for document in self.corpus.documents:
            reasons = self._rejection_reasons(query, document, set(), check_conflict=False)
            if not reasons and not document.exception_of:
                by_family[document.family].append(document)
        conflicts: list[str] = []
        for family, documents in by_family.items():
            same_precedence: dict[tuple[Any, int], list[PolicyDocument]] = defaultdict(list)
            for document in documents:
                same_precedence[(document.effective_from, document.priority)].append(document)
            for same_level in same_precedence.values():
                if len(same_level) > 1 and len({item.text for item in same_level}) > 1:
                    ids = ",".join(sorted(item.document_id for item in same_level))
                    conflicts.append(f"POLICY_VERSION_CONFLICT:{family}:{ids}")
        return sorted(set(conflicts))

    @staticmethod
    def _rejection_reasons(
        query: RetrievalQuery,
        document: PolicyDocument,
        conflict_families: set[str],
        *,
        check_conflict: bool = True,
    ) -> list[str]:
        reasons: list[str] = []
        if document.status != "published":
            reasons.append("NOT_PUBLISHED")
        if document.effective_from > query.as_of:
            reasons.append("NOT_YET_EFFECTIVE")
        if document.effective_to is not None and query.as_of > document.effective_to:
            reasons.append("EXPIRED")
        product_type = query.metadata_filter.get("product_type")
        if product_type and str(product_type) not in document.tags:
            reasons.append("PRODUCT_TYPE_MISMATCH")
        reason = query.metadata_filter.get("reason")
        required_reason_tag = {
            "quality_issue": "quality",
            "no_reason_return": "no_reason_return",
        }.get(str(reason))
        if required_reason_tag and required_reason_tag not in document.tags:
            reasons.append("AFTER_SALES_REASON_MISMATCH")
        required_scopes = {
            str(scope) for scope in document.metadata.get("required_scopes", [])
        }
        if required_scopes and not required_scopes.issubset(query.permission_scopes):
            reasons.append("PERMISSION_DENIED")
        version = query.metadata_filter.get("version")
        if version and str(version) != document.version:
            reasons.append("VERSION_MISMATCH")
        if check_conflict and document.family in conflict_families:
            reasons.append("ACTIVE_VERSION_CONFLICT")
        return reasons

