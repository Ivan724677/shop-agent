"""Load individual JSON scenarios and generated JSONL corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .scenario_schema import Scenario


def load_file(path: Path) -> Scenario:
    with path.open(encoding="utf-8") as handle:
        return Scenario.from_dict(json.load(handle))


def load_jsonl(path: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                scenarios.append(Scenario.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{line_number} 场景格式错误：{error}") from error
    return scenarios


def load_directory(directory: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for path in sorted(directory.glob("*.json")):
        scenarios.append(load_file(path))
    for path in sorted(directory.glob("*.jsonl")):
        scenarios.extend(load_jsonl(path))
    return scenarios


def load_corpus(root: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for directory in (root / "golden", root / "generated", root / "fault_injection"):
        if directory.exists():
            scenarios.extend(load_directory(directory))
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("场景 ID 必须全局唯一。")
    return scenarios


def summarize(scenarios: Iterable[Scenario]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for scenario in scenarios:
        for tag in scenario.tags or ["untagged"]:
            summary[tag] = summary.get(tag, 0) + 1
    return summary
