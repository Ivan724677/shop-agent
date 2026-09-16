"""Online/shadow evaluation over production RAG telemetry and feedback."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from rag.models import AgenticRAGResult
from rag.monitoring import RAGMonitor, RAGTelemetry


@dataclass(frozen=True)
class OnlineEvaluationSample:
    run_id: str
    status: str
    user_feedback: str | None = None
    expected_status: str | None = None
    expected_evidence_ids: tuple[str, ...] = ()
    actual_evidence_ids: tuple[str, ...] = ()
    actual_citation_ids: tuple[str, ...] = ()


class OnlineRAGEvaluator:
    """Rolling evaluator for live traffic, shadow labels and delayed feedback."""

    def __init__(self, monitor: RAGMonitor | None = None) -> None:
        self.monitor = monitor or RAGMonitor()
        self.samples: list[OnlineEvaluationSample] = []

    def record_result(
        self,
        result: AgenticRAGResult,
        *,
        user_feedback: str | None = None,
        expected_status: str | None = None,
        expected_evidence_ids: list[str] | None = None,
    ) -> None:
        self.samples.append(
            OnlineEvaluationSample(
                run_id=result.run_id,
                status=result.status,
                user_feedback=user_feedback,
                expected_status=expected_status,
                expected_evidence_ids=tuple(expected_evidence_ids or []),
                actual_evidence_ids=tuple(result.evidence_ids),
                actual_citation_ids=tuple(result.citation_ids),
            )
        )

    def snapshot(self) -> dict[str, Any]:
        output = self.monitor.snapshot()
        if not self.samples:
            output["labeled_samples"] = 0
            return output
        labeled = [sample for sample in self.samples if sample.expected_status]
        evidence_labeled = [sample for sample in self.samples if sample.expected_evidence_ids]
        output["labeled_samples"] = len(labeled)
        output["status_accuracy"] = (
            sum(sample.status == sample.expected_status for sample in labeled) / len(labeled)
            if labeled else None
        )
        output["evidence_recall"] = (
            sum(
                set(sample.expected_evidence_ids).issubset(
                    set(sample.actual_evidence_ids) | set(sample.actual_citation_ids)
                )
                for sample in evidence_labeled
            )
            / len(evidence_labeled)
            if evidence_labeled else None
        )
        output["feedback_counts"] = dict(Counter(sample.user_feedback for sample in self.samples if sample.user_feedback))
        return output
