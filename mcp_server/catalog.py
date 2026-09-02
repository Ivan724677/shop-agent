"""Canonical MCP tool catalog and model-facing schema adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required_scopes: frozenset[str]
    risk_level: str = "low"
    side_effect: bool = False
    inject_user_id: bool = False
    timeout_seconds: float = 1.0
    max_attempts: int = 2
    idempotency_field: str | None = None
    reconciliation_tool: str | None = None

    def as_mcp_tool(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "outputSchema": self.output_schema,
            "annotations": {
                "readOnlyHint": not self.side_effect,
                "destructiveHint": self.risk_level == "high",
                "idempotentHint": self.idempotency_field is not None,
            },
        }

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def _object(
    properties: dict[str, Any], required: list[str], *, additional: bool = False
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": additional,
    }


def _output(required_key: str) -> dict[str, Any]:
    return _object({required_key: {}}, [required_key], additional=True)


ORDER_ID = {"type": "string", "pattern": r"^O\d{5,}$"}
ITEM_ID = {"type": "string", "pattern": r"^I\d{3,}$"}
PRODUCT_ID = {"type": "string", "pattern": r"^P\d{3,}$"}
SHIPMENT_ID = {"type": "string", "minLength": 1, "maxLength": 64}
REFUND_ID = {"type": "string", "pattern": r"^RF\d{5,}$"}
REASON = {"type": "string", "enum": ["no_reason_return", "quality_issue"]}
IDEMPOTENCY_KEY = {
    "type": "string",
    "minLength": 8,
    "maxLength": 200,
    "description": "同一业务操作必须稳定复用，禁止放入用户隐私。",
}


MCP_TOOLS: dict[str, ToolDefinition] = {
    tool.name: tool
    for tool in [
        ToolDefinition(
            "get_user",
            "查询当前已认证用户资料。用户身份由服务端会话注入。",
            _object({}, []),
            _output("user"),
            frozenset({"profile:read"}),
            inject_user_id=True,
        ),
        ToolDefinition(
            "list_orders",
            "列出当前已认证用户的订单。",
            _object({}, []),
            _output("orders"),
            frozenset({"orders:read"}),
            inject_user_id=True,
        ),
        ToolDefinition(
            "get_order",
            "按订单号查询当前用户的订单及订单项。",
            _object({"order_id": ORDER_ID}, ["order_id"]),
            _output("order"),
            frozenset({"orders:read"}),
            inject_user_id=True,
        ),
        ToolDefinition(
            "get_product",
            "按商品目录 ID 查询商品当前信息。",
            _object({"product_id": PRODUCT_ID}, ["product_id"]),
            _output("product"),
            frozenset({"catalog:read"}),
        ),
        ToolDefinition(
            "list_shipments",
            "查询当前用户一个订单关联的全部物流包裹。",
            _object({"order_id": ORDER_ID}, ["order_id"]),
            _output("shipments"),
            frozenset({"shipments:read"}),
            inject_user_id=True,
        ),
        ToolDefinition(
            "get_shipment",
            "查询当前用户指定物流包裹及完整轨迹。",
            _object({"shipment_id": SHIPMENT_ID}, ["shipment_id"]),
            _output("shipment"),
            frozenset({"shipments:read"}),
            inject_user_id=True,
        ),
        ToolDefinition(
            "search_policy",
            "为当前用户订单中的指定商品检索适用售后政策证据。",
            _object(
                {"order_id": ORDER_ID, "item_id": ITEM_ID, "reason": REASON},
                ["order_id", "item_id", "reason"],
            ),
            _output("evidence"),
            frozenset({"policy:read", "orders:read"}),
            inject_user_id=True,
        ),
        ToolDefinition(
            "calculate_refund",
            "校验售后资格并计算退款金额，不执行写操作。",
            _object(
                {
                    "order_id": ORDER_ID,
                    "item_ids": {
                        "type": "array",
                        "items": ITEM_ID,
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                    "reason": REASON,
                },
                ["order_id", "item_ids", "reason"],
            ),
            _object(
                {
                    "eligible_items": {"type": "array"},
                    "ineligible_items": {"type": "array"},
                    "refund_amount": {"type": "number", "minimum": 0},
                },
                ["eligible_items", "ineligible_items", "refund_amount"],
                additional=True,
            ),
            frozenset({"returns:quote", "orders:read"}),
            risk_level="medium",
            inject_user_id=True,
        ),
        ToolDefinition(
            "create_return_request",
            "创建退货申请。必须有用户确认、服务端金额复核和稳定幂等键。",
            _object(
                {
                    "order_id": ORDER_ID,
                    "item_ids": {
                        "type": "array",
                        "items": ITEM_ID,
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                    "refund_amount": {"type": "number", "exclusiveMinimum": 0},
                    "reason": REASON,
                    "idempotency_key": IDEMPOTENCY_KEY,
                    "user_confirmation": {"type": "boolean"},
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
            _output("request_id"),
            frozenset({"returns:write", "orders:read"}),
            risk_level="high",
            side_effect=True,
            inject_user_id=True,
            max_attempts=1,
            idempotency_field="idempotency_key",
            reconciliation_tool="get_return_status",
        ),
        ToolDefinition(
            "get_return_status",
            "使用原幂等键查询当前用户的退货申请是否已经创建。",
            _object({"idempotency_key": IDEMPOTENCY_KEY}, ["idempotency_key"]),
            _object(
                {"found": {"type": "boolean"}, "request": {}},
                ["found", "request"],
                additional=True,
            ),
            frozenset({"returns:read"}),
            inject_user_id=True,
        ),
        ToolDefinition(
            "get_refund_status",
            "按退款流水号查询当前用户的退款状态。",
            _object({"refund_id": REFUND_ID}, ["refund_id"]),
            _output("refund"),
            frozenset({"refunds:read"}),
            inject_user_id=True,
        ),
        ToolDefinition(
            "create_ticket",
            "无法自动处理时创建幂等的人工工单。",
            _object(
                {
                    "subject": {"type": "string", "minLength": 2, "maxLength": 100},
                    "description": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 2000,
                        "x-sensitive": True,
                    },
                    "order_id": ORDER_ID,
                    "idempotency_key": IDEMPOTENCY_KEY,
                },
                ["subject", "description", "idempotency_key"],
            ),
            _output("ticket"),
            frozenset({"tickets:write"}),
            risk_level="medium",
            side_effect=True,
            inject_user_id=True,
            max_attempts=1,
            idempotency_field="idempotency_key",
            reconciliation_tool="get_ticket_status",
        ),
        ToolDefinition(
            "get_ticket_status",
            "使用幂等键查询当前用户的人工工单是否已经创建。",
            _object({"idempotency_key": IDEMPOTENCY_KEY}, ["idempotency_key"]),
            _object(
                {"found": {"type": "boolean"}, "ticket": {}},
                ["found", "ticket"],
                additional=True,
            ),
            frozenset({"tickets:read"}),
            inject_user_id=True,
        ),
    ]
}


OPENAI_TOOLS = [definition.as_openai_tool() for definition in MCP_TOOLS.values()]

DEFAULT_CUSTOMER_SCOPES = frozenset(
    scope for definition in MCP_TOOLS.values() for scope in definition.required_scopes
)
