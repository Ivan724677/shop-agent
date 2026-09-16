"""Optional LLM tie-breaker used only when deterministic routing is ambiguous."""

from __future__ import annotations

import json
from typing import Any, Protocol

from baseline.deepseek_client import ChatModelClient

from .models import ExpertName, SemanticRouteCandidate


ROUTE_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_route_decision",
        "description": "为复杂客服请求给出专家候选；不能授权任何业务写操作。",
        "parameters": {
            "type": "object",
            "properties": {
                "experts": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [expert.value for expert in ExpertName],
                    },
                    "uniqueItems": True,
                    "maxItems": 4,
                },
                "needs_clarification": {"type": "boolean"},
                "reason": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["experts", "needs_clarification", "reason", "confidence"],
            "additionalProperties": False,
        },
    },
}


ROUTER_SYSTEM_PROMPT = """你是电商售后系统的路由候选生成器，不是执行 Agent。
结构化状态和确定性规则拥有最终决定权。你只在多意图、冲突或低置信度时，
从 order_logistics、policy、transaction、handoff 中给出候选，或要求澄清。
transaction 只是候选，绝不表示用户已确认，也不能绕过政策检查。必须调用函数。"""


class SemanticRouter(Protocol):
    def route(
        self,
        user_message: str,
        state_snapshot: dict[str, Any],
        rule_experts: list[ExpertName],
        complexity_reasons: list[str],
    ) -> SemanticRouteCandidate:
        ...


class EmptySemanticRouter:
    def route(
        self,
        user_message: str,
        state_snapshot: dict[str, Any],
        rule_experts: list[ExpertName],
        complexity_reasons: list[str],
    ) -> SemanticRouteCandidate:
        return SemanticRouteCandidate()


class DeepSeekSemanticRouter:
    def __init__(self, model_client: ChatModelClient) -> None:
        self.model_client = model_client
        self.last_usage: dict[str, Any] = {}
        self.usage_history: list[dict[str, Any]] = []

    def route(
        self,
        user_message: str,
        state_snapshot: dict[str, Any],
        rule_experts: list[ExpertName],
        complexity_reasons: list[str],
    ) -> SemanticRouteCandidate:
        payload = {
            "user_message": user_message,
            "structured_state": state_snapshot,
            "rule_experts": [expert.value for expert in rule_experts],
            "complexity_reasons": complexity_reasons,
        }
        completion = self.model_client.complete(
            [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            [ROUTE_DECISION_TOOL],
        )
        self.last_usage = dict(completion.usage)
        self.usage_history.append(self.last_usage)
        raw = self._extract_arguments(completion.message)
        if not isinstance(raw.get("experts"), list):
            raise ValueError("语义 Router 的 experts 必须是数组。")
        if not isinstance(raw.get("needs_clarification"), bool):
            raise ValueError("语义 Router 的 needs_clarification 必须是布尔值。")
        if not isinstance(raw.get("reason"), str):
            raise ValueError("语义 Router 的 reason 必须是字符串。")
        if isinstance(raw.get("confidence"), bool) or not isinstance(
            raw.get("confidence"), (int, float)
        ):
            raise ValueError("语义 Router 的 confidence 必须是数字。")
        candidates: list[ExpertName] = []
        for value in raw.get("experts", []):
            try:
                expert = ExpertName(value)
            except (TypeError, ValueError):
                continue
            if expert not in candidates:
                candidates.append(expert)
        confidence = float(raw["confidence"])
        return SemanticRouteCandidate(
            experts=candidates,
            needs_clarification=raw["needs_clarification"],
            reason=raw["reason"],
            confidence=max(0.0, min(confidence, 1.0)),
            usage=self.last_usage,
        )

    @staticmethod
    def _extract_arguments(message: dict[str, Any]) -> dict[str, Any]:
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("语义 Router 必须返回且只返回一个 emit_route_decision 调用。")
        function = calls[0].get("function") if isinstance(calls[0], dict) else None
        if not isinstance(function, dict) or function.get("name") != "emit_route_decision":
            raise ValueError("语义 Router 返回了不允许的函数。")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ValueError("语义 Router 返回了无效 JSON 参数。") from exc
        if not isinstance(arguments, dict):
            raise ValueError("语义 Router 参数必须是 JSON 对象。")
        return arguments
