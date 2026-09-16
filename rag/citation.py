"""Claim-level citation and grounding validation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .generation import GeneratedAnswer
from .models import EvidenceValidation


@dataclass(frozen=True)
class CitationValidation:
    accepted: bool
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "invalid_citations": list(self.invalid_citations),
            "unsupported_claims": list(self.unsupported_claims),
            "missing_citations": list(self.missing_citations),
            "reason_codes": list(self.reason_codes),
        }


class CitationValidator:
    """Conservative deterministic gate after LLM generation.

    This is intentionally not an LLM judge. It verifies citation identity,
    claim/evidence lexical support and numeric consistency. A production system
    may add a calibrated entailment model, but it must remain behind this gate.
    """

    def validate(self, generated: GeneratedAnswer, evidence: EvidenceValidation) -> CitationValidation:
        valid = {item.citation_id for item in evidence.accepted}
        evidence_text = {item.citation_id: item.document.text for item in evidence.accepted}
        invalid: list[str] = []
        unsupported: list[str] = []
        missing: list[str] = []
        if generated.abstain:
            return CitationValidation(False, reason_codes=["GENERATOR_ABSTAINED"])
        for claim in generated.claims:
            if not claim.citation_ids:
                missing.append(claim.text)
                continue
            unknown = [citation for citation in claim.citation_ids if citation not in valid]
            invalid.extend(unknown)
            if unknown:
                continue
            sources = " ".join(evidence_text[citation] for citation in claim.citation_ids)
            if not _claim_supported(claim.text, sources):
                unsupported.append(claim.text)
        reasons: list[str] = []
        if invalid:
            reasons.append("INVALID_CITATION_ID")
        if missing:
            reasons.append("CLAIM_CITATION_MISSING")
        if unsupported:
            reasons.append("UNSUPPORTED_CLAIM")
        if not generated.claims:
            reasons.append("NO_CLAIMS")
        return CitationValidation(
            accepted=not reasons,
            invalid_citations=sorted(set(invalid)),
            unsupported_claims=unsupported,
            missing_citations=missing,
            reason_codes=reasons or ["CITATIONS_VERIFIED"],
        )


def _claim_supported(claim: str, source: str) -> bool:
    claim_numbers = _numeric_facts(claim)
    source_numbers = _numeric_facts(source)
    if not claim_numbers.issubset(source_numbers):
        return False
    claim_terms = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", claim.lower()))
    source_terms = set(re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", source.lower()))
    if not claim_terms:
        return False
    overlap = len(claim_terms & source_terms) / len(claim_terms)
    return overlap >= 0.35


_NUMERIC_FACT_PATTERN = re.compile(
    r"(?:"
    r"\d+(?:\.\d+)?(?:个工作日|工作日|天|日|小时|分钟|元|折|%|％|次|件|个月|月|年)?"
    r"|[零〇一二两三四五六七八九十百千万]+(?:个工作日|工作日|天|日|小时|分钟|元|折|次|件|个月|月|年)"
    r")"
)


def _numeric_facts(text: str) -> set[str]:
    """Extract numbers together with units, including common Chinese numerals."""

    return {match.group(0) for match in _NUMERIC_FACT_PATTERN.finditer(text)}
