"""OpenAI-compatible tool schemas and a session-bound execution adapter."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from agent import PolicyKnowledge, ToolGateway, ToolResult
from repositories import InMemoryStore


def _function(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


REASON_SCHEMA = {
    "type": "string",
    "enum": ["no_reason_return", "quality_issue"],
    "description": "售后原因：无理由退货或质量问题。",
}


BASELINE_TOOLS = [
    _function("get_user", "查询当前会话用户资料。", {}, []),
    _function("list_orders", "列出当前用户的所有订单。", {}, []),
    _function(
        "get_order",
        "按订单号查询当前用户的订单及订单项。",
        {"order_id": {"type": "string", "description": "订单号，例如 O10086。"}},
        ["order_id"],
    ),
    _function(
        "get_product",
        "按商品目录 ID 查询商品当前信息。",
        {"product_id": {"type": "string", "description": "商品目录 ID，例如 P002。"}},
        ["product_id"],
    ),
    _function(
        "list_shipments",
        "查询一个订单关联的全部物流包裹。",
        {"order_id": {"type": "string", "description": "订单号。"}},
        ["order_id"],
    ),
    _function(
        "get_shipment",
        "查询指定物流包裹及完整轨迹。",
        {"shipment_id": {"type": "string", "description": "物流记录 ID。"}},
        ["shipment_id"],
    ),
    _function(
        "search_policy",
        "为订单中的指定商品检索适用售后政策证据。",
        {
            "order_id": {"type": "string", "description": "订单号。"},
            "item_id": {"type": "string", "description": "订单项 ID。"},
            "reason": REASON_SCHEMA,
        },
        ["order_id", "item_id", "reason"],
    ),
    _function(
        "calculate_refund",
        "校验商品售后资格并计算退款金额；不会执行写操作。",
        {
            "order_id": {"type": "string", "description": "订单号。"},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "需要处理的订单项 ID 列表。",
            },
            "reason": REASON_SCHEMA,
        },
        ["order_id", "item_ids", "reason"],
    ),
    _function(
        "create_return_request",
        "创建退货申请。仅在用户明确确认具体订单、商品和金额后调用。",
        {
            "order_id": {"type": "string", "description": "订单号。"},
            "item_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "用户确认退货的订单项 ID。",
            },
            "refund_amount": {"type": "number", "description": "calculate_refund 返回的金额。"},
            "reason": REASON_SCHEMA,
            "idempotency_key": {
                "type": "string",
                "description": "本次写操作的稳定唯一键；重试同一操作时必须复用。",
            },
            "user_confirmation": {
                "type": "boolean",
                "description": "用户是否已明确确认该次退货操作。",
            },
        },
        [
            "order_id",
            "item_ids",
            "refund_amount",
            "reason",
            "idempotency_key",
            "user_confirmation",
        ],
    ),
    _function(
        "get_return_status",
        "使用原幂等键查询退货申请是否已经创建。",
        {"idempotency_key": {"type": "string", "description": "原写操作的幂等键。"}},
        ["idempotency_key"],
    ),
    _function(
        "get_refund_status",
        "按退款流水号查询退款状态。",
        {"refund_id": {"type": "string", "description": "退款流水号，例如 RF00001。"}},
        ["refund_id"],
    ),
    _function(
        "create_ticket",
        "无法自动处理时创建人工工单。",
        {
            "subject": {"type": "string", "description": "简短工单标题。"},
            "description": {"type": "string", "description": "需要人工处理的事实和原因。"},
            "order_id": {"type": "string", "description": "可选关联订单号。"},
        },
        ["subject", "description"],
    ),
]


SESSION_SCOPED_TOOLS = {
    "get_user",
    "list_orders",
    "get_order",
    "list_shipments",
    "get_shipment",
    "search_policy",
    "calculate_refund",
    "create_return_request",
    "get_refund_status",
    "create_ticket",
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def tool_result_as_json(result: ToolResult) -> str:
    payload = {
        "ok": result.ok,
        "status": result.status,
        "data": _jsonable(result.data),
        "error": (
            {
                "code": result.error_code,
                "message": result.message,
                "retryable": result.retryable,
            }
            if not result.ok
            else None
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class SessionToolExecutor:
    """Bind authenticated user identity outside model-controlled arguments."""

    def __init__(
        self,
        user_id: str = "U001",
        store: InMemoryStore | None = None,
        gateway: ToolGateway | None = None,
    ) -> None:
        self.user_id = user_id
        if gateway is not None:
            self.gateway = gateway
            self.store = store or gateway.store
        else:
            self.store = store or InMemoryStore()
            self.gateway = ToolGateway(self.store, PolicyKnowledge())
        self.allowed_tool_names = {
            tool["function"]["name"] for tool in BASELINE_TOOLS
        }

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_name not in self.allowed_tool_names:
            return ToolResult(
                ok=False,
                status="ERROR",
                error_code="UNKNOWN_TOOL",
                message=f"baseline 不允许调用工具 {tool_name}。",
            )
        if not isinstance(arguments, dict):
            return ToolResult(
                ok=False,
                status="VALIDATION_ERROR",
                error_code="ARGUMENTS_NOT_OBJECT",
                message="工具参数必须是 JSON 对象。",
            )

        safe_arguments = dict(arguments)
        claimed_user_id = safe_arguments.pop("user_id", None)
        if claimed_user_id is not None and claimed_user_id != self.user_id:
            return ToolResult(
                ok=False,
                status="FORBIDDEN",
                error_code="IDENTITY_OVERRIDE_ATTEMPT",
                message="模型不能覆盖当前会话的用户身份。",
            )
        if tool_name in SESSION_SCOPED_TOOLS:
            safe_arguments["user_id"] = self.user_id
        try:
            return self.gateway.call(tool_name, **safe_arguments)
        except (TypeError, ValueError) as exc:
            return ToolResult(
                ok=False,
                status="VALIDATION_ERROR",
                error_code="TOOL_ARGUMENT_MISMATCH",
                message=f"工具参数类型或字段不匹配：{exc}",
            )
