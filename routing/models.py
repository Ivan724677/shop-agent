"""Typed contracts shared by the stage-five router and expert agents."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ExpertName(StringEnum):
    ORDER_LOGISTICS = "order_logistics"
    POLICY = "policy"
    TRANSACTION = "transaction"
    HANDOFF = "handoff"


class ExpertStatus(StringEnum):
    COMPLETE = "complete"
    CONTINUE = "continue"
    CLARIFY = "clarify"
    REJECTED = "rejected"
    ESCALATE = "escalate"


class RouteSource(StringEnum):
    RULE = "rule"
    RULE_WITH_MODEL = "rule_with_model"
    MODEL = "model"
    FALLBACK = "fallback"


@dataclass
class ExpertResult:
    expert: ExpertName
    status: ExpertStatus
    response: str = ""
    next_expert: ExpertName | None = None
    reason_code: str = ""
    escalation_reason: str = ""
    facts: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouteDecision:
    experts: list[ExpertName]
    reason_codes: list[str]
    source: RouteSource
    model_invoked: bool = False
    model_usage: dict[str, Any] = field(default_factory=dict)
    model_error: str | None = None
    model_candidates: list[ExpertName] = field(default_factory=list)
    model_reason: str = ""
    model_confidence: float = 0.0
    clarification_message: str = ""
    route_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "experts": [expert.value for expert in self.experts],
            "reason_codes": list(self.reason_codes),
            "source": self.source.value,
            "model_invoked": self.model_invoked,
            "model_usage": dict(self.model_usage),
            "model_error": self.model_error,
            "model_candidates": [expert.value for expert in self.model_candidates],
            "model_reason": self.model_reason,
            "model_confidence": self.model_confidence,
            "clarification_message": self.clarification_message,
        }


@dataclass
class SemanticRouteCandidate:
    experts: list[ExpertName] = field(default_factory=list)
    needs_clarification: bool = False
    reason: str = ""
    confidence: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)
