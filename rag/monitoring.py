"""Operational telemetry and monitor-agent style anomaly detection."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RAGTelemetry:
    run_id: str
    status: str
    stop_reason: str
    retrieval_count: int
    planner_calls: int
    generator_calls: int
    latency_ms: float
    evidence_count: int
    accepted_evidence_count: int
    citation_valid: bool | None
    model_error: str | None
    evidence_ids: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RAGMonitor:
    """In-process collector; replace sink with OTEL/Kafka in deployment."""

    def __init__(self, sink: Path | None = None, *, max_samples: int = 10000) -> None:
        self.sink = Path(sink) if sink else None
        self.max_samples = max_samples
        self.samples: list[RAGTelemetry] = []
        self._lock = threading.RLock()

    def record(self, telemetry: RAGTelemetry) -> None:
        with self._lock:
            self.samples.append(telemetry)
            if len(self.samples) > self.max_samples:
                self.samples = self.samples[-self.max_samples :]
            if self.sink:
                self.sink.parent.mkdir(parents=True, exist_ok=True)
                with self.sink.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(telemetry.as_dict(), ensure_ascii=False) + "\n")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            samples = list(self.samples)
        if not samples:
            return {"samples": 0}
        statuses = Counter(sample.status for sample in samples)
        alerts = Counter(alert for sample in samples for alert in sample.alerts)
        return {
            "samples": len(samples),
            "status_counts": dict(statuses),
            "average_latency_ms": sum(sample.latency_ms for sample in samples) / len(samples),
            "average_retrieval_count": sum(sample.retrieval_count for sample in samples) / len(samples),
            "average_planner_calls": sum(sample.planner_calls for sample in samples) / len(samples),
            "grounded_rate": sum(sample.status == "GROUNDED" for sample in samples) / len(samples),
            "uncertain_rate": sum(sample.status == "UNCERTAIN" for sample in samples) / len(samples),
            "alert_counts": dict(alerts),
        }


class MonitorAgent:
    """Rule-based first responder that can later be replaced by an LLM analyst."""

    def analyze(self, telemetry: RAGTelemetry) -> list[str]:
        alerts = list(telemetry.alerts)
        if telemetry.status == "GROUNDED" and telemetry.accepted_evidence_count == 0:
            alerts.append("GROUNDED_WITHOUT_ACCEPTED_EVIDENCE")
        if telemetry.status == "GROUNDED" and telemetry.citation_valid is False:
            alerts.append("GROUNDED_WITH_INVALID_CITATION")
        if telemetry.retrieval_count > 3:
            alerts.append("EXCESSIVE_RETRIEVAL_STEPS")
        if telemetry.model_error:
            alerts.append("MODEL_OR_PROVIDER_ERROR")
        return sorted(set(alerts))


def new_run_id() -> str:
    return str(uuid.uuid4())
