"""Summarize production RAG telemetry with delayed/shadow labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rag.monitoring import RAGMonitor, RAGTelemetry

from .online_rag import OnlineEvaluationSample, OnlineRAGEvaluator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate production RAG telemetry")
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"telemetry 第 {line_number} 行不是合法 JSON。") from exc
        if not isinstance(row, dict):
            raise ValueError(f"telemetry 第 {line_number} 行必须是对象。")
        rows.append(row)
    return rows


def _read_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        rows = [dict(value, run_id=key) for key, value in raw.items() if isinstance(value, dict)]
    else:
        raise ValueError("labels 顶层必须是数组或 run_id 到标签对象的映射。")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("run_id"):
            raise ValueError("每条 label 必须包含 run_id。")
        output[str(row["run_id"])] = row
    return output


def evaluate(telemetry_rows: list[dict[str, Any]], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    monitor = RAGMonitor()
    evaluator = OnlineRAGEvaluator(monitor)
    for row in telemetry_rows:
        telemetry = RAGTelemetry(
            run_id=str(row.get("run_id", "")),
            status=str(row.get("status", "UNKNOWN")),
            stop_reason=str(row.get("stop_reason", "")),
            retrieval_count=int(row.get("retrieval_count", 0)),
            planner_calls=int(row.get("planner_calls", 0)),
            generator_calls=int(row.get("generator_calls", 0)),
            latency_ms=float(row.get("latency_ms", 0.0)),
            evidence_count=int(row.get("evidence_count", 0)),
            accepted_evidence_count=int(row.get("accepted_evidence_count", 0)),
            citation_valid=row.get("citation_valid"),
            model_error=row.get("model_error"),
            evidence_ids=[str(value) for value in row.get("evidence_ids", [])],
            citation_ids=[str(value) for value in row.get("citation_ids", [])],
            alerts=[str(value) for value in row.get("alerts", [])],
            timestamp=float(row.get("timestamp", 0.0)),
        )
        monitor.record(telemetry)
        label = labels.get(telemetry.run_id, {})
        evaluator.samples.append(
            OnlineEvaluationSample(
                run_id=telemetry.run_id,
                status=telemetry.status,
                expected_status=label.get("expected_status"),
                expected_evidence_ids=tuple(str(value) for value in label.get("expected_evidence_ids", [])),
                actual_evidence_ids=tuple(telemetry.evidence_ids),
                actual_citation_ids=tuple(telemetry.citation_ids),
                user_feedback=label.get("user_feedback"),
            )
        )
    snapshot = evaluator.snapshot()
    latencies = sorted(float(row.get("latency_ms", 0.0)) for row in telemetry_rows)
    snapshot["p95_latency_ms"] = (
        latencies[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)]
        if latencies else None
    )
    snapshot["alert_rate"] = (
        sum(bool(row.get("alerts")) for row in telemetry_rows) / len(telemetry_rows)
        if telemetry_rows else None
    )
    snapshot["unlabeled_samples"] = sum(row.get("run_id") not in labels for row in telemetry_rows)
    return snapshot


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate(_read_jsonl(args.telemetry), _read_labels(args.labels))
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
