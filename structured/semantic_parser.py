"""LLM 语义解析器（阶段三感知层的模型分支）。

让 DeepSeek 把用户消息解析成受约束的 SemanticFrame 候选：
模型只能调用单一函数 emit_semantic_frame（JSON Schema 约束字段与枚举），
只接收最小 parser_context（不接收原始聊天历史）。

解析结果只是候选，无任何执行权限，最终由 StateReducer 定权。
EmptySemanticParser 用于 --offline 离线模式（纯确定性提取器也能跑通基本流程）。
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from baseline.deepseek_client import ChatModelClient

from .models import SemanticFrame


SEMANTIC_FRAME_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_semantic_frame",
        "description": "将当前用户消息解析为客服任务语义帧。必须调用此函数。",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "unknown",
                        "keep_current",
                        "query_order",
                        "query_logistics",
                        "request_return",
                        "human_handoff",
                    ],
                },
                "order_id": {"type": "string"},
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "product_mentions": {"type": "array", "items": {"type": "string"}},
                "excluded_item_ids": {"type": "array", "items": {"type": "string"}},
                "excluded_product_mentions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "reason": {
                    "type": "string",
                    "enum": ["no_reason_return", "quality_issue"],
                },
                "consultation_only": {"type": "boolean"},
                "confirmation": {
                    "type": "string",
                    "enum": ["none", "confirm", "reject"],
                },
                "wants_handoff": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "intent",
                "item_ids",
                "product_mentions",
                "excluded_item_ids",
                "excluded_product_mentions",
                "consultation_only",
                "confirmation",
                "wants_handoff",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
}


PARSER_SYSTEM_PROMPT = """你是电商售后对话的语义解析器，不是执行 Agent。
只解析当前用户消息，并结合给出的最小任务上下文解决“它、上一单、确认”等指代。
不要决定工具调用，不要声称业务操作已完成。必须调用 emit_semantic_frame。
当消息只是对当前待确认动作说“确认/取消”时，intent 使用 keep_current。
consultation_only 表示用户只咨询资格或金额，并未要求现在执行。"""


class SemanticParser(Protocol):
    def parse(self, user_message: str, state_context: dict[str, Any]) -> SemanticFrame:
        ...


class EmptySemanticParser:
    """Offline fallback: deterministic extraction still keeps basic flows usable."""

    def parse(self, user_message: str, state_context: dict[str, Any]) -> SemanticFrame:
        return SemanticFrame()


class DeepSeekSemanticParser:
    def __init__(self, model_client: ChatModelClient) -> None:
        self.model_client = model_client
        self.usage_history: list[dict[str, Any]] = []
        self.last_usage: dict[str, Any] = {}

    def parse(self, user_message: str, state_context: dict[str, Any]) -> SemanticFrame:
        content = json.dumps(
            {"state_context": state_context, "user_message": user_message},
            ensure_ascii=False,
        )
        completion = self.model_client.complete(
            [
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            [SEMANTIC_FRAME_TOOL],
        )
        self.last_usage = dict(completion.usage)
        self.usage_history.append(self.last_usage)
        raw = self._extract_arguments(completion.message)
        return SemanticFrame.from_dict(raw)

    @staticmethod
    def _extract_arguments(message: dict[str, Any]) -> dict[str, Any]:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict) or function.get("name") != "emit_semantic_frame":
                    continue
                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    return arguments
                if isinstance(arguments, str):
                    try:
                        parsed = json.loads(arguments)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed

        content = message.get("content")
        if isinstance(content, str):
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].lstrip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                raise ValueError("语义解析模型没有返回合法的结构化帧。") from exc
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("语义解析模型没有调用 emit_semantic_frame。")
