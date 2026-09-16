"""Agentic RAG planning: model chooses the next retrieval action, not facts."""

from __future__ import annotations

import json
from typing import Any, Protocol

from baseline.deepseek_client import ChatModelClient

from .models import PlanAction, PlanDecision


RAG_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_rag_plan",
        "description": "选择政策知识库 Agent 的下一步；不能直接给出未验证政策结论。",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [action.value for action in PlanAction],
                },
                "reason": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "query": {"type": "string"},
                "retrieval_mode": {"type": "string", "enum": ["vector", "hybrid"]},
                "metadata_filter": {"type": "object"},
            },
            "required": ["action", "reason", "confidence", "query", "retrieval_mode", "metadata_filter"],
            "additionalProperties": False,
        },
    },
}


RAG_PLANNER_PROMPT = """你是电商售后政策知识库 Agent 的规划器。
你不负责直接回答事实，只负责选择下一步：
- answer_directly：问题不需要政策知识库，或已有充分已验证证据；
- retrieve：查询知识库；
- rewrite_query：证据不足时重写查询并再次查询；
- verify：要求系统基于现有证据做版本、适用范围和冲突校验；
- clarify：用户信息不足；
- escalate：证据冲突或无法安全判断。
你必须根据当前观察结果逐步决策，不要假设检索结果中不存在的事实。
政策资格、例外和退款期限问题没有通过证据校验时不能 answer_directly。
必须调用 emit_rag_plan。"""


class RAGPlanner(Protocol):
    def plan(self, observation: dict[str, Any]) -> PlanDecision:
        ...


class HeuristicRAGPlanner:
    """Offline planner used for deterministic replay; online mode uses DeepSeek."""

    POLICY_WORDS = (
        "退货",
        "退款",
        "售后",
        "政策",
        "规则",
        "能不能退",
        "可以退",
        "例外",
        "定制",
        "质量",
        "几天",
        "期限",
    )

    def plan(self, observation: dict[str, Any]) -> PlanDecision:
        if observation.get("evidence_sufficient"):
            return PlanDecision(
                action=PlanAction.ANSWER_DIRECTLY,
                reason="已有通过校验的证据，继续检索收益低",
                confidence=0.95,
            )
        if observation.get("has_conflict"):
            return PlanDecision(
                action=PlanAction.ESCALATE,
                reason="发现有效政策版本冲突，不能猜测采用哪条",
                confidence=0.99,
            )
        query = str(observation.get("user_query", ""))
        if not observation.get("policy_sensitive"):
            return PlanDecision(
                action=PlanAction.ANSWER_DIRECTLY,
                reason="当前问题不涉及政策事实",
                confidence=0.98,
            )
        if observation.get("retrieval_count", 0) == 0:
            return PlanDecision(
                action=PlanAction.RETRIEVE,
                reason="问题涉及退货资格或售后政策，需要证据",
                confidence=0.9,
                query=query,
                retrieval_mode="hybrid",
                metadata_filter=dict(observation.get("metadata_filter", {})),
            )
        if observation.get("rewritten"):
            return PlanDecision(
                action=PlanAction.VERIFY,
                reason="已完成二次检索，进入证据校验",
                confidence=0.9,
            )
        return PlanDecision(
            action=PlanAction.REWRITE_QUERY,
            reason="初始证据不足，补充商品类型、原因和期限维度",
            confidence=0.85,
            query=query,
            metadata_filter=dict(observation.get("metadata_filter", {})),
        )


class DeepSeekRAGPlanner:
    def __init__(self, model_client: ChatModelClient) -> None:
        self.model_client = model_client
        self.last_usage: dict[str, Any] = {}
        self.usage_history: list[dict[str, Any]] = []

    def plan(self, observation: dict[str, Any]) -> PlanDecision:
        completion = self.model_client.complete(
            [
                {"role": "system", "content": RAG_PLANNER_PROMPT},
                {"role": "user", "content": json.dumps(observation, ensure_ascii=False)},
            ],
            [RAG_PLAN_TOOL],
        )
        self.last_usage = dict(completion.usage)
        self.usage_history.append(self.last_usage)
        raw = self._extract_arguments(completion.message)
        try:
            action = PlanAction(raw["action"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("RAG Planner 返回了无效 action。") from exc
        if not isinstance(raw.get("reason"), str) or not isinstance(raw.get("query"), str):
            raise ValueError("RAG Planner 的 reason/query 必须是字符串。")
        if not isinstance(raw.get("metadata_filter"), dict):
            raise ValueError("RAG Planner 的 metadata_filter 必须是对象。")
        if raw.get("retrieval_mode") not in {"vector", "hybrid"}:
            raise ValueError("RAG Planner 的 retrieval_mode 无效。")
        if isinstance(raw.get("confidence"), bool) or not isinstance(raw.get("confidence"), (int, float)):
            raise ValueError("RAG Planner 的 confidence 必须是数字。")
        return PlanDecision(
            action=action,
            reason=raw["reason"],
            confidence=max(0.0, min(float(raw["confidence"]), 1.0)),
            query=raw["query"],
            retrieval_mode=raw["retrieval_mode"],
            metadata_filter=dict(raw["metadata_filter"]),
            usage=self.last_usage,
        )

    @staticmethod
    def _extract_arguments(message: dict[str, Any]) -> dict[str, Any]:
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("RAG Planner 必须返回且只返回一个函数调用。")
        function = calls[0].get("function") if isinstance(calls[0], dict) else None
        if not isinstance(function, dict) or function.get("name") != "emit_rag_plan":
            raise ValueError("RAG Planner 返回了不允许的函数。")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("RAG Planner 参数必须是 JSON 对象。")
        return arguments
