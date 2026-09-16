"""Evidence validation: applicability, precedence, coverage and conflicts."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from .models import EvidenceValidation, PolicyDocument, RetrievalQuery, RetrievedEvidence
from .retrieval import PolicyCorpus


def _version_number(version: str) -> int:
    match = re.search(r"(\d+)", version)
    return int(match.group(1)) if match else 0


class EvidenceValidator:
    def __init__(self, corpus: PolicyCorpus) -> None:
        self.corpus = corpus

    def validate(
        self,
        query: RetrievalQuery,
        evidence: list[RetrievedEvidence],
        required_aspects: list[str] | None = None,
    ) -> EvidenceValidation:
        required = list(required_aspects or self._required_aspects(query))
        applicable = self.corpus.filter(query)
        by_family: dict[str, list[PolicyDocument]] = defaultdict(list)
        for document in applicable:
            by_family[document.family].append(document)

        conflicts: list[str] = []
        chosen: dict[str, set[str]] = {}
        for family, documents in by_family.items():
            # Exceptions are deliberate overrides, not version conflicts. If
            # two base rules share the same precedence and overlap, stop rather
            # than silently selecting the first vector hit.
            bases = [document for document in documents if not document.exception_of]
            precedence = defaultdict(list)
            for document in bases:
                precedence[(document.effective_from, document.priority)].append(document)
            for key, same_level in precedence.items():
                if len(same_level) > 1 and len({document.text for document in same_level}) > 1:
                    ids = ",".join(sorted(document.document_id for document in same_level))
                    conflicts.append(f"POLICY_VERSION_CONFLICT:{family}:{ids}")
            max_priority = max((document.priority for document in documents), default=0)
            priority_docs = [document for document in documents if document.priority == max_priority]
            latest_date = max((document.effective_from for document in priority_docs), default=date.min)
            latest = [document for document in priority_docs if document.effective_from == latest_date]
            max_version = max((_version_number(document.version) for document in latest), default=0)
            chosen[family] = {
                document.document_id
                for document in latest
                if _version_number(document.version) == max_version
            }

        accepted: list[RetrievedEvidence] = []
        rejected: list[str] = []
        low_relevance_rejected = False
        stale_or_precedence_rejected = False
        for item in evidence:
            document = item.document
            # A metadata match alone is not evidence.  A document returned with
            # zero lexical/vector signal is a candidate for debugging, but it
            # must not become a grounding source merely because the filter was
            # narrow.  This is what lets Agentic RAG notice that its first
            # query was semantically weak and decide to rewrite it.
            if max(item.vector_score, item.lexical_score, item.rerank_score) <= 0:
                rejected.append(document.document_id)
                low_relevance_rejected = True
                continue
            if not document.is_active(query.as_of):
                rejected.append(document.document_id)
                stale_or_precedence_rejected = True
                continue
            if conflicts and document.family in {key.split(":")[1] for key in conflicts}:
                rejected.append(document.document_id)
                continue
            if document.document_id not in chosen.get(document.family, set()):
                rejected.append(document.document_id)
                stale_or_precedence_rejected = True
                continue
            accepted.append(item)

        missing: list[str] = []
        accepted_tags = set().union(*(item.document.tags for item in accepted)) if accepted else set()
        for aspect in required:
            if aspect not in accepted_tags and not any(aspect in item.document.text for item in accepted):
                missing.append(aspect)
        reasons: list[str] = []
        if conflicts:
            reasons.append("CONFLICTING_ACTIVE_POLICIES")
        if stale_or_precedence_rejected:
            reasons.append("STALE_OR_LOWER_PRECEDENCE_EVIDENCE_REJECTED")
        if low_relevance_rejected:
            reasons.append("LOW_RELEVANCE_EVIDENCE_REJECTED")
        if missing:
            reasons.append("EVIDENCE_COVERAGE_INCOMPLETE")
        if accepted and not conflicts and not missing:
            reasons.append("EVIDENCE_VERIFIED")
        return EvidenceValidation(
            accepted=accepted,
            rejected_ids=sorted(set(rejected)),
            conflicts=sorted(set(conflicts)),
            missing_aspects=sorted(set(missing)),
            reason_codes=reasons,
        )

    @staticmethod
    def _required_aspects(query: RetrievalQuery) -> list[str]:
        aspects: list[str] = []
        if query.metadata_filter.get("product_type"):
            aspects.append(str(query.metadata_filter["product_type"]))
        if query.metadata_filter.get("reason") == "quality_issue":
            aspects.append("quality")
        if any(word in query.text for word in ("多少", "几天", "期限", "时间")):
            aspects.append("return")
        return list(dict.fromkeys(aspects))
