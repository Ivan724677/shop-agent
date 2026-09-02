"""风险感知售后客服 Agent 的最小可行版本。

这个版本刻意不依赖任何大模型或第三方 SDK：先用规则模拟意图解析，
把真正容易出问题的工程部分（任务状态、工具契约、政策证据、幂等、
故障恢复和 trace 监控）做成可运行、可测试的骨架。

后续接入 LLM 时，只需替换 RuleBasedNLU 和回复生成层；交易工具的
安全边界不应交给模型决定。
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Optional

from domain.models import Order, OrderItem, Product, Shipment
from repositories.in_memory import InMemoryStore, ToolError


TODAY = date(2026, 7, 16)  # 固定日期使测试和演示结果可复现。


@dataclass
class ToolResult:
    ok: bool
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    message: str = ""
    retryable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceEvent:
    trace_id: str
    timestamp: str
    tool: str
    risk_level: str
    input: dict[str, Any]
    ok: bool
    status: str
    error_code: Optional[str] = None


@dataclass
class TaskState:
    """结构化工作记忆：高风险操作只能依据这里的字段执行。"""

    session_id: str
    intent: Optional[str] = None
    order_id: Optional[str] = None
    selected_item_ids: set[str] = field(default_factory=set)
    excluded_item_ids: set[str] = field(default_factory=set)
    reason: str = "no_reason_return"
    policy_evidence: list[str] = field(default_factory=list)
    refund_amount: Optional[float] = None
    awaiting_confirmation: bool = False
    completed: bool = False
    last_response: str = ""


class PolicyKnowledge:
    """可替换为向量库+重排器的最小政策检索层，并返回可审计的证据。"""

    def __init__(self, documents: Optional[list[dict[str, Any]]] = None) -> None:
        if documents is None:
            policy_path = Path(__file__).parent / "seed" / "data" / "policies.json"
            with policy_path.open(encoding="utf-8") as handle:
                documents = json.load(handle)
        self.documents = [
            {**document, "tags": set(document.get("tags", []))} for document in documents
        ]

    def search(self, item: OrderItem, reason: str) -> list[dict[str, str]]:
        wanted_tags = {"return"}
        wanted_tags.add("custom" if item.is_custom else "standard")
        if reason == "quality_issue":
            wanted_tags.add("quality")

        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in self.documents:
            score = len(wanted_tags.intersection(doc["tags"]))
            if score:
                scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {"id": doc["id"], "title": doc["title"], "text": doc["text"]}
            for _, doc in scored[:2]
        ]


@dataclass
class ToolSpec:
    name: str
    risk_level: str
    required_fields: set[str]
    handler: Callable[..., dict[str, Any]]
    side_effect: bool = False


class ToolGateway:
    """MCP 风格的工具网关：统一校验、故障注入、幂等恢复和 trace。"""

    def __init__(self, store: InMemoryStore, knowledge: PolicyKnowledge) -> None:
        self.store = store
        self.knowledge = knowledge
        self.trace_id = str(uuid.uuid4())
        self.trace: list[TraceEvent] = []
        self.fail_next: dict[str, list[str]] = {}
        self.tools: dict[str, ToolSpec] = {
            "get_user": ToolSpec("get_user", "low", {"user_id"}, self._get_user),
            "get_product": ToolSpec("get_product", "low", {"product_id"}, self._get_product),
            "list_orders": ToolSpec("list_orders", "low", {"user_id"}, self._list_orders),
            "get_order": ToolSpec("get_order", "low", {"user_id", "order_id"}, self._get_order),
            "list_shipments": ToolSpec(
                "list_shipments", "low", {"user_id", "order_id"}, self._list_shipments
            ),
            "get_shipment": ToolSpec(
                "get_shipment", "low", {"user_id", "shipment_id"}, self._get_shipment
            ),
            "search_policy": ToolSpec(
                "search_policy",
                "low",
                {"user_id", "order_id", "item_id", "reason"},
                self._search_policy,
            ),
            "calculate_refund": ToolSpec(
                "calculate_refund",
                "medium",
                {"user_id", "order_id", "item_ids", "reason"},
                self._calculate_refund,
            ),
            "create_return_request": ToolSpec(
                "create_return_request",
                "high",
                {
                    "user_id",
                    "order_id",
                    "item_ids",
                    "refund_amount",
                    "reason",
                    "idempotency_key",
                    "user_confirmation",
                },
                self._create_return_request,
                side_effect=True,
            ),
            "get_return_status": ToolSpec(
                "get_return_status",
                "low",
                {"user_id", "idempotency_key"},
                self._get_return_status,
            ),
            "get_refund_status": ToolSpec(
                "get_refund_status", "low", {"user_id", "refund_id"}, self._get_refund_status
            ),
            "create_ticket": ToolSpec(
                "create_ticket",
                "medium",
                {"user_id", "subject", "description"},
                self._create_ticket,
                side_effect=True,
            ),
            "get_ticket_status": ToolSpec(
                "get_ticket_status",
                "low",
                {"user_id", "idempotency_key"},
                self._get_ticket_status,
            ),
        }

    def inject_failure_once(self, tool_name: str, failure_mode: str) -> None:
        """failure_mode 可为 TIMEOUT 或 TIMEOUT_AFTER_COMMIT。"""
        if tool_name not in self.tools:
            raise ValueError(f"未知工具：{tool_name}")
        self.fail_next.setdefault(tool_name, []).append(failure_mode)

    def inject_failures(self, tool_name: str, *failure_modes: str) -> None:
        for failure_mode in failure_modes:
            self.inject_failure_once(tool_name, failure_mode)

    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        if tool_name not in self.tools:
            return ToolResult(False, "ERROR", error_code="UNKNOWN_TOOL", message="工具不存在。")
        spec = self.tools[tool_name]
        missing = sorted(spec.required_fields - kwargs.keys())
        if missing:
            result = ToolResult(
                False,
                "VALIDATION_ERROR",
                error_code="MISSING_REQUIRED_FIELDS",
                message=f"工具参数缺失：{', '.join(missing)}",
            )
            self._record(spec, kwargs, result)
            return result

        failures = self.fail_next.get(tool_name, [])
        failure_mode = failures.pop(0) if failures else None
        if not failures:
            self.fail_next.pop(tool_name, None)
        try:
            if failure_mode == "TIMEOUT":
                result = ToolResult(
                    False,
                    "TIMEOUT",
                    error_code="TOOL_TIMEOUT",
                    message="工具调用超时，操作没有提交。",
                    retryable=True,
                )
            elif failure_mode == "TIMEOUT_AFTER_COMMIT":
                # 模拟最危险的情况：服务端已经写入，但网络响应丢失。
                spec.handler(**kwargs)
                result = ToolResult(
                    False,
                    "UNKNOWN_COMMIT",
                    error_code="UNKNOWN_COMMIT",
                    message="请求响应丢失，操作是否提交未知；禁止直接重复写入。",
                    retryable=False,
                )
            else:
                result = ToolResult(True, "OK", data=spec.handler(**kwargs))
        except ToolError as error:
            result = ToolResult(
                False,
                "ERROR",
                error_code=error.code,
                message=str(error),
                retryable=error.retryable,
            )

        self._record(spec, kwargs, result)
        return result

    def _record(self, spec: ToolSpec, tool_input: dict[str, Any], result: ToolResult) -> None:
        self.trace.append(
            TraceEvent(
                trace_id=self.trace_id,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                tool=spec.name,
                risk_level=spec.risk_level,
                input=tool_input,
                ok=result.ok,
                status=result.status,
                error_code=result.error_code,
            )
        )

    @staticmethod
    def _serialize_order(order: Order) -> dict[str, Any]:
        return {
            "order_id": order.order_id,
            "user_id": order.user_id,
            "status": str(order.status),
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "payment_method": order.payment_method,
            "items": [
                {
                    "item_id": item.item_id,
                    "product_id": item.product_id,
                    "name": item.name,
                    "price": item.price,
                    "quantity": item.quantity,
                    "category": item.category,
                    "delivered_on": item.delivered_on.isoformat() if item.delivered_on else None,
                    "is_custom": item.is_custom,
                    "return_status": str(item.return_status),
                    "shipment_id": item.shipment_id,
                }
                for item in order.items
            ],
        }

    @staticmethod
    def _serialize_user(user: Any) -> dict[str, Any]:
        return {
            "user_id": user.user_id,
            "name": user.name,
            "phone": user.phone,
            "email": user.email,
            "membership_level": str(user.membership_level),
            "status": str(user.status),
            "addresses": [
                {
                    "address_id": address.address_id,
                    "recipient": address.recipient,
                    "phone": address.phone,
                    "province": address.province,
                    "city": address.city,
                    "district": address.district,
                    "detail": address.detail,
                    "is_default": address.is_default,
                }
                for address in user.addresses
            ],
        }

    @staticmethod
    def _serialize_product(product: Product) -> dict[str, Any]:
        return {
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "current_price": product.current_price,
            "is_custom": product.is_custom,
            "return_tags": sorted(product.return_tags),
            "active": product.active,
        }

    @staticmethod
    def _serialize_shipment(shipment: Shipment) -> dict[str, Any]:
        return {
            "shipment_id": shipment.shipment_id,
            "order_id": shipment.order_id,
            "tracking_number": shipment.tracking_number,
            "carrier": shipment.carrier,
            "status": str(shipment.status),
            "item_ids": shipment.item_ids,
            "shipped_at": shipment.shipped_at.isoformat() if shipment.shipped_at else None,
            "estimated_delivery": (
                shipment.estimated_delivery.isoformat() if shipment.estimated_delivery else None
            ),
            "delivered_on": shipment.delivered_on.isoformat() if shipment.delivered_on else None,
            "events": [
                {
                    "event_id": event.event_id,
                    "status": str(event.status),
                    "occurred_at": event.occurred_at.isoformat(),
                    "location": event.location,
                    "description": event.description,
                }
                for event in shipment.events
            ],
        }

    @staticmethod
    def _serialize_return(request: Any) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "order_id": request.order_id,
            "item_ids": request.item_ids,
            "refund_amount": request.refund_amount,
            "reason": request.reason,
            "status": str(request.status),
            "idempotency_key": request.idempotency_key,
            "created_at": request.created_at.isoformat(),
            "timeline": [asdict(change) for change in request.timeline],
        }

    @staticmethod
    def _serialize_refund(refund: Any) -> dict[str, Any]:
        return {
            "refund_id": refund.refund_id,
            "user_id": refund.user_id,
            "order_id": refund.order_id,
            "request_id": refund.request_id,
            "item_ids": refund.item_ids,
            "amount": refund.amount,
            "method": str(refund.method),
            "status": str(refund.status),
            "created_at": refund.created_at.isoformat(),
            "settled_at": refund.settled_at.isoformat() if refund.settled_at else None,
            "timeline": [asdict(change) for change in refund.timeline],
        }

    def _list_orders(self, user_id: str) -> dict[str, Any]:
        return {"orders": [self._serialize_order(order) for order in self.store.list_orders(user_id)]}

    def _get_user(self, user_id: str) -> dict[str, Any]:
        return {"user": self._serialize_user(self.store.get_user(user_id))}

    def _get_product(self, product_id: str) -> dict[str, Any]:
        return {"product": self._serialize_product(self.store.get_product(product_id))}

    def _get_order(self, user_id: str, order_id: str) -> dict[str, Any]:
        return {"order": self._serialize_order(self.store.get_order(user_id, order_id))}

    def _list_shipments(self, user_id: str, order_id: str) -> dict[str, Any]:
        return {
            "shipments": [
                self._serialize_shipment(shipment)
                for shipment in self.store.list_shipments(user_id, order_id)
            ]
        }

    def _get_shipment(self, user_id: str, shipment_id: str) -> dict[str, Any]:
        return {"shipment": self._serialize_shipment(self.store.get_shipment(user_id, shipment_id))}

    def _search_policy(
        self, user_id: str, order_id: str, item_id: str, reason: str
    ) -> dict[str, Any]:
        order = self.store.get_order(user_id, order_id)
        item = next((item for item in order.items if item.item_id == item_id), None)
        if item is None:
            raise ToolError("ITEM_NOT_IN_ORDER", "商品不属于该订单。")
        return {"evidence": self.knowledge.search(item, reason)}

    def _calculate_refund(
        self, user_id: str, order_id: str, item_ids: list[str], reason: str
    ) -> dict[str, Any]:
        order = self.store.get_order(user_id, order_id)
        items_by_id = {item.item_id: item for item in order.items}
        eligible: list[dict[str, Any]] = []
        ineligible: list[dict[str, str]] = []
        for item_id in item_ids:
            item = items_by_id.get(item_id)
            if item is None:
                raise ToolError("ITEM_NOT_IN_ORDER", "存在不属于该订单的商品。")
            allowed, reason_text = self.store.check_eligibility(item, reason)
            if allowed:
                eligible.append({"item_id": item.item_id, "name": item.name, "amount": item.price})
            else:
                ineligible.append({"item_id": item.item_id, "name": item.name, "reason": reason_text})
        return {
            "eligible_items": eligible,
            "ineligible_items": ineligible,
            "refund_amount": round(sum(item["amount"] for item in eligible), 2),
        }

    def _create_return_request(self, **kwargs: Any) -> dict[str, Any]:
        return self._serialize_return(self.store.create_return_request(**kwargs))

    def _get_return_status(self, user_id: str, idempotency_key: str) -> dict[str, Any]:
        request = self.store.get_return_by_idempotency(user_id, idempotency_key)
        return {
            "found": request is not None,
            "request": self._serialize_return(request) if request else None,
        }

    def _get_refund_status(self, user_id: str, refund_id: str) -> dict[str, Any]:
        refund = self.store.refund_transactions.get(refund_id)
        if refund is None:
            raise ToolError("REFUND_NOT_FOUND", f"退款流水 {refund_id} 不存在。")
        if refund.user_id != user_id:
            raise ToolError("FORBIDDEN", "当前用户无权访问该退款流水。")
        return {"refund": self._serialize_refund(refund)}

    def _create_ticket(self, **kwargs: Any) -> dict[str, Any]:
        ticket = self.store.create_ticket(**kwargs)
        return {"ticket": {
            "ticket_id": ticket.ticket_id,
            "user_id": ticket.user_id,
            "order_id": ticket.order_id,
            "subject": ticket.subject,
            "description": ticket.description,
            "status": str(ticket.status),
            "priority": str(ticket.priority),
            "type": str(ticket.ticket_type),
            "created_at": ticket.created_at.isoformat(),
        }}

    def _get_ticket_status(self, user_id: str, idempotency_key: str) -> dict[str, Any]:
        ticket = self.store.get_ticket_by_idempotency(user_id, idempotency_key)
        return {
            "found": ticket is not None,
            "ticket": (
                {
                    "ticket_id": ticket.ticket_id,
                    "user_id": ticket.user_id,
                    "order_id": ticket.order_id,
                    "subject": ticket.subject,
                    "status": str(ticket.status),
                }
                if ticket
                else None
            ),
        }


class RuleBasedNLU:
    """MVP 的可解释 NLU；接口可直接替换成结构化 LLM 输出。"""

    PRODUCT_HINTS = {"鞋": "I002", "耳机": "I001", "马克杯": "I003", "杯": "I003"}

    @classmethod
    def parse(cls, text: str) -> dict[str, Any]:
        normalized = text.replace(" ", "")
        order_match = re.search(r"O\d{5,}", normalized, flags=re.IGNORECASE)
        order_id = order_match.group(0).upper() if order_match else None
        targets: set[str] = set()
        excluded: set[str] = set()
        for hint, item_id in cls.PRODUCT_HINTS.items():
            if hint in normalized:
                targets.add(item_id)
                # 不能用“商品名后任意五个字符”判断排除：
                # “退鞋，耳机不退”会错误地把鞋也排除。这里要求否定词
                # 与商品名直接相邻，避免跨商品作用域污染。
                if re.search(
                    fr"(?:{hint}(?:不退|不要退|别退)|(?:不退|不要退|别退|保留){hint})",
                    normalized,
                ):
                    excluded.add(item_id)

        is_return = any(keyword in normalized for keyword in ("退货", "退款", "退掉", "退"))
        is_query = any(keyword in normalized for keyword in ("查询", "查一下", "物流", "订单状态", "到哪"))
        confirmation = bool(
            re.search(r"^(确认|确定|同意|执行|好的|好吧|是的|可以)$", normalized)
            or "确认退货" in normalized
            or "确认退款" in normalized
        )
        if is_return:
            intent = "request_return"
        elif is_query:
            intent = "query_order"
        else:
            intent = None
        return {
            "intent": intent,
            "order_id": order_id,
            "selected_item_ids": targets - excluded,
            "excluded_item_ids": excluded,
            "reason": "quality_issue"
            if any(keyword in normalized for keyword in ("质量", "坏了", "损坏", "故障"))
            else "no_reason_return",
            "confirmation": confirmation,
        }


class MonitorAgent:
    """基于 trace 的在线监控：只给出告警，绝不自动修改交易状态。"""

    def inspect(self, trace: list[TraceEvent]) -> list[dict[str, str]]:
        alerts: list[dict[str, str]] = []
        for event in trace:
            if event.status == "UNKNOWN_COMMIT":
                alerts.append(
                    {
                        "severity": "critical",
                        "type": "unknown_commit",
                        "message": "高风险写操作响应丢失；系统必须先查询状态，不能盲目重试。",
                    }
                )
            if event.tool == "create_return_request" and not event.input.get("user_confirmation"):
                alerts.append(
                    {
                        "severity": "critical",
                        "type": "missing_confirmation",
                        "message": "检测到未确认的高风险写操作。",
                    }
                )
            if event.tool == "search_policy" and event.ok is False:
                alerts.append(
                    {
                        "severity": "warning",
                        "type": "policy_retrieval_failure",
                        "message": "政策检索失败，应转人工或要求澄清，禁止自动退款。",
                    }
                )
        return alerts


class CustomerServiceAgent:
    """订单 Agent、政策 Agent、交易 Agent 的最小编排器。"""

    def __init__(self, user_id: str = "U001") -> None:
        self.user_id = user_id
        self.store = InMemoryStore()
        self.gateway = ToolGateway(self.store, PolicyKnowledge())
        self.state = TaskState(session_id=str(uuid.uuid4()))
        self.monitor = MonitorAgent()

    def handle(self, user_message: str) -> str:
        parsed = RuleBasedNLU.parse(user_message)
        if self.state.completed and parsed["intent"] == "request_return":
            self.state = TaskState(session_id=str(uuid.uuid4()))
        self._merge_into_state(parsed)

        if self.state.intent == "request_return":
            response = self._handle_return()
        elif self.state.intent == "query_order":
            response = self._handle_order_query()
        else:
            response = (
                "我可以帮你查询订单、物流或申请退货。"
                "请提供订单号，例如：我要退订单 O10086 里的鞋。"
            )
        self.state.last_response = response
        return response

    def _merge_into_state(self, parsed: dict[str, Any]) -> None:
        if parsed["intent"]:
            self.state.intent = parsed["intent"]
        if parsed["order_id"]:
            self.state.order_id = parsed["order_id"]
        if parsed["selected_item_ids"]:
            self.state.selected_item_ids.update(parsed["selected_item_ids"])
        if parsed["excluded_item_ids"]:
            self.state.excluded_item_ids.update(parsed["excluded_item_ids"])
            self.state.selected_item_ids.difference_update(parsed["excluded_item_ids"])
        if parsed["reason"] == "quality_issue":
            self.state.reason = "quality_issue"
        if parsed["confirmation"]:
            self.state.awaiting_confirmation = False
            setattr(self.state, "confirmed", True)

    def _handle_order_query(self) -> str:
        if not self.state.order_id:
            result = self.gateway.call("list_orders", user_id=self.user_id)
            order_ids = [order["order_id"] for order in result.data["orders"]]
            return f"我找到了你的订单：{'、'.join(order_ids)}。请告诉我需要查询哪一单。"
        result = self.gateway.call("get_order", user_id=self.user_id, order_id=self.state.order_id)
        if not result.ok:
            return f"查询失败：{result.message}"
        order = result.data["order"]
        item_text = "；".join(
            f"{item['name']}（{'已签收' if item['delivered_on'] else '运输中'}）"
            for item in order["items"]
        )
        return f"订单 {order['order_id']} 当前状态为 {order['status']}：{item_text}。"

    def _handle_return(self) -> str:
        if not self.state.order_id:
            return "为避免退错订单，请先提供订单号，例如 O10086。"

        order_result = self.gateway.call("get_order", user_id=self.user_id, order_id=self.state.order_id)
        if not order_result.ok:
            return f"无法继续处理：{order_result.message}"
        order = self.store.get_order(self.user_id, self.state.order_id)

        if not self.state.selected_item_ids:
            choices = "、".join(f"{item.name}（{item.item_id}）" for item in order.items)
            return f"订单 {order.order_id} 包含：{choices}。请明确告诉我需要退哪一件商品。"

        valid_item_ids = {item.item_id for item in order.items}
        invalid_item_ids = self.state.selected_item_ids - valid_item_ids
        if invalid_item_ids:
            return "选中的商品不属于当前订单。为安全起见，请重新确认订单号和商品。"

        selected_items = [item for item in order.items if item.item_id in self.state.selected_item_ids]
        evidence_ids: list[str] = []
        for item in selected_items:
            policy_result = self.gateway.call(
                "search_policy",
                user_id=self.user_id,
                order_id=order.order_id,
                item_id=item.item_id,
                reason=self.state.reason,
            )
            if not policy_result.ok:
                return "政策检索异常，当前不能自动创建退货申请，已建议转人工处理。"
            evidence_ids.extend(item["id"] for item in policy_result.data["evidence"])
        self.state.policy_evidence = sorted(set(evidence_ids))

        calculation = self.gateway.call(
            "calculate_refund",
            user_id=self.user_id,
            order_id=order.order_id,
            item_ids=sorted(self.state.selected_item_ids),
            reason=self.state.reason,
        )
        if not calculation.ok:
            return f"退款金额计算失败：{calculation.message}"
        ineligible = calculation.data["ineligible_items"]
        if ineligible:
            details = "；".join(item["reason"] for item in ineligible)
            return f"当前不能自动退货：{details}"

        amount = calculation.data["refund_amount"]
        self.state.refund_amount = amount
        item_names = "、".join(item.name for item in selected_items)
        confirmed = getattr(self.state, "confirmed", False)
        if not confirmed:
            self.state.awaiting_confirmation = True
            return (
                f"已核验订单 {order.order_id}：将只为 {item_names} 创建退货申请，"
                f"预计退款 {amount:.2f} 元；不会处理你排除的其他商品。"
                "如需继续，请回复“确认退货”。"
            )

        idempotency_key = (
            f"{self.state.session_id}:{order.order_id}:"
            f"{','.join(sorted(self.state.selected_item_ids))}"
        )
        creation = self.gateway.call(
            "create_return_request",
            user_id=self.user_id,
            order_id=order.order_id,
            item_ids=sorted(self.state.selected_item_ids),
            refund_amount=amount,
            reason=self.state.reason,
            idempotency_key=idempotency_key,
            user_confirmation=True,
        )
        if creation.status == "UNKNOWN_COMMIT":
            # 绝不直接重试写操作，先以幂等键做状态对账。
            reconciliation = self.gateway.call(
                "get_return_status",
                user_id=self.user_id,
                idempotency_key=idempotency_key,
            )
            request = reconciliation.data.get("request") if reconciliation.ok else None
            if request:
                self.state.completed = True
                return (
                    f"退款接口响应丢失，但已通过幂等键对账确认："
                    f"退货申请 {request['request_id']} 已创建，退款 {request['refund_amount']:.2f} 元。"
                )
            return "退货操作状态不明，系统没有重试写入，已建议转人工核实。"
        if not creation.ok:
            return f"创建退货申请失败：{creation.message}"

        self.state.completed = True
        request = creation.data
        return (
            f"退货申请 {request['request_id']} 已创建：{item_names}，"
            f"退款金额 {request['refund_amount']:.2f} 元。"
        )

    def trace_as_json(self) -> str:
        return json.dumps([asdict(event) for event in self.gateway.trace], ensure_ascii=False, indent=2)

    def monitor_alerts(self) -> list[dict[str, str]]:
        return self.monitor.inspect(self.gateway.trace)


def run_demo(inject_timeout: bool = False) -> None:
    agent = CustomerServiceAgent()
    if inject_timeout:
        agent.gateway.inject_failure_once("create_return_request", "TIMEOUT_AFTER_COMMIT")

    messages = ["我要退订单 O10086 里的鞋，耳机不退", "确认退货"]
    print("=== 风险感知售后 Agent 演示 ===")
    for message in messages:
        print(f"用户：{message}")
        print(f"Agent：{agent.handle(message)}")
    print("\n=== Trace ===")
    print(agent.trace_as_json())
    print("\n=== Monitor 告警 ===")
    alerts = agent.monitor_alerts()
    print(json.dumps(alerts, ensure_ascii=False, indent=2) if alerts else "无告警")


def run_cli() -> None:
    agent = CustomerServiceAgent()
    print("风险感知售后客服 Agent。输入 exit 结束；输入 trace 查看执行轨迹。")
    while True:
        try:
            message = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n会话结束。")
            return
        if message.lower() in {"exit", "quit", "退出"}:
            print("会话结束。")
            return
        if message.lower() == "trace":
            print(agent.trace_as_json())
            print("监控告警：", json.dumps(agent.monitor_alerts(), ensure_ascii=False))
            continue
        if message:
            print("Agent：" + agent.handle(message))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="风险感知售后客服 Agent MVP")
    parser.add_argument("--demo", action="store_true", help="运行部分退货演示")
    parser.add_argument(
        "--inject-timeout",
        action="store_true",
        help="演示高风险写操作提交后响应丢失的恢复流程",
    )
    args = parser.parse_args()
    if args.demo:
        run_demo(inject_timeout=args.inject_timeout)
    else:
        run_cli()
