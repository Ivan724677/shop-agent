"""Schema and validation for replayable customer-service scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    scenario_id: str
    description: str
    user_id: str
    conversation: list[dict[str, str]]
    business_state: dict[str, Any]
    expected_intent: str
    required_slots: list[str]
    allowed_actions: list[str]
    forbidden_actions: list[str]
    expected_final_state: str
    risk_level: str
    fault_injection: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Scenario":
        scenario = cls(**raw)
        scenario.validate()
        return scenario

    def validate(self) -> None:
        required = {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "user_id": self.user_id,
            "expected_intent": self.expected_intent,
            "expected_final_state": self.expected_final_state,
            "risk_level": self.risk_level,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"场景缺少必填字段：{', '.join(missing)}")
        if not self.conversation:
            raise ValueError(f"场景 {self.scenario_id} 必须至少有一轮对话。")
        if self.risk_level not in {"low", "medium", "high", "critical"}:
            raise ValueError(f"场景 {self.scenario_id} 的风险等级无效。")
        if set(self.allowed_actions) & set(self.forbidden_actions):
            raise ValueError(f"场景 {self.scenario_id} 的允许和禁止动作有交集。")

