"""Small Prometheus text exporter without a process-global SDK dependency."""

from __future__ import annotations

import threading
from collections import defaultdict

from .config import settings


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def render(self) -> str:
        namespace = settings.metrics_namespace.replace("-", "_")
        lines: list[str] = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                metric = f"{namespace}_{name}"
                lines.extend([f"# TYPE {metric} counter", f"{metric} {value}"])
            for name, value in sorted(self._gauges.items()):
                metric = f"{namespace}_{name}"
                lines.extend([f"# TYPE {metric} gauge", f"{metric} {value}"])
        return "\n".join(lines) + "\n"


metrics = MetricsRegistry()

