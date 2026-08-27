"""确定性对话策略（阶段三决策层第二环，负责"行动决策"）。

读取 StructuredTaskState，决定当前该做什么：
- 意图路由：未知→追问；查询→只读工具；退货→完整退货流程；人工→建工单
- 退货流程：查订单→解析商品→检索政策→计算金额→冻结 PendingAction
- 确认后 scope_key 复核，一致才执行 create_return_request
- UNKNOWN_COMMIT：不重试写操作，先幂等对账，查不到则转人工
- 高风险工具失败：创建工单转人工，不盲目重试

本层不 import 工具定义，通过注入的 call_tool 函数调工具，与执行层解耦。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent import ToolResult

from .models import (
    ConfirmationStatus,
    Intent,
    PendingAction,
    RiskLevel,
    SlotSource,
    StructuredTaskState,
    TaskStage,
    ToolFact,
)
from .reducer import StateReducer
from .state_machine import transition_state


ToolCaller = Callable[[str, dict], ToolResult]


@dataclass
class PolicyOutcome:
    response: str
    final_state: TaskStage


class DialoguePolicy:
    def __init__(self, tool_caller: ToolCaller) -> None:
        self.call_tool = tool_caller

    def decide(self, state: StructuredTaskState) -> PolicyOutcome:
        if state.confirmation_status == ConfirmationStatus.REJECTED:
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            state.risk_level = RiskLevel.MEDIUM
            return PolicyOutcome("已取消刚才待确认的操作，没有创建退货申请。", state.stage)
        if state.intent == Intent.HUMAN_HANDOFF:
            return self._handoff(state)
        if state.intent == Intent.UNKNOWN:
            transition_state(state, TaskStage.COLLECTING_INFO)
            state.missing_fields = ["intent"]
            return PolicyOutcome(
                "我可以查询订单和物流，也可以处理退货。请告诉我具体想办理什么。",
                state.stage,
            )
        if state.intent in {Intent.QUERY_ORDER, Intent.QUERY_LOGISTICS}:
            return self._query(state)
        if state.intent == Intent.REQUEST_RETURN:
            return self._return_flow(state)
        transition_state(state, TaskStage.COLLECTING_INFO)
        return PolicyOutcome("请补充订单号和具体问题。", state.stage)

    def _query(self, state: StructuredTaskState) -> PolicyOutcome:
        if not state.order_id:
            result = self._observe(state, "list_orders", {})
            if not result.ok:
                return self._tool_failure(state, result, "订单列表查询失败")
            order_ids = [order["order_id"] for order in result.data.get("orders", [])]
            state.missing_fields = ["order_id"]
            transition_state(state, TaskStage.COLLECTING_INFO)
            choices = "、".join(order_ids) if order_ids else "没有可用订单"
            return PolicyOutcome(
                f"我找到了这些订单：{choices}。请明确告诉我需要查询哪一单。",
                state.stage,
            )

        transition_state(state, TaskStage.RESOLVING_ENTITIES)
        order_result = self._observe(state, "get_order", {"order_id": state.order_id})
        if not order_result.ok:
            return self._tool_failure(state, order_result, "订单查询失败")
        order = order_result.data["order"]

        transition_state(state, TaskStage.EXECUTING)
        if state.intent == Intent.QUERY_LOGISTICS:
            shipment_result = self._observe(
                state, "list_shipments", {"order_id": state.order_id}
            )
            if not shipment_result.ok:
                return self._tool_failure(state, shipment_result, "物流查询失败")
            shipments = shipment_result.data.get("shipments", [])
            parts = [
                f"{shipment['carrier']} {shipment['tracking_number']}：{shipment['status']}"
                for shipment in shipments
            ]
            response = (
                f"订单 {state.order_id} 有 {len(shipments)} 个包裹：" + "；".join(parts) + "。"
            )
        else:
            items = "、".join(
                f"{item['name']}（{item['item_id']}）" for item in order.get("items", [])
            )
            response = f"订单 {state.order_id} 状态为 {order['status']}，包含：{items}。"
        state.missing_fields = []
        transition_state(state, TaskStage.COMPLETED)
        return PolicyOutcome(response, state.stage)

    def _return_flow(self, state: StructuredTaskState) -> PolicyOutcome:
        if not state.order_id:
            transition_state(state, TaskStage.COLLECTING_INFO)
            state.missing_fields = ["order_id"]
            return PolicyOutcome("为避免处理错订单，请先提供订单号。", state.stage)

        if state.stage != TaskStage.AWAITING_CONFIRMATION:
            transition_state(state, TaskStage.RESOLVING_ENTITIES)
        order_result = self._observe(state, "get_order", {"order_id": state.order_id})
        if not order_result.ok:
            return self._tool_failure(state, order_result, "订单查询失败")
        order = order_result.data["order"]
        resolved, excluded, unresolved = self._resolve_items(state, order.get("items", []))
        if unresolved:
            state.conflicts.append("UNRESOLVED_PRODUCT_MENTION:" + ",".join(unresolved))
        if not resolved:
            state.selected_item_ids = []
            state.missing_fields = ["target_item"]
            transition_state(state, TaskStage.COLLECTING_INFO)
            choices = "、".join(
                f"{item['name']}（{item['item_id']}）" for item in order.get("items", [])
            )
            return PolicyOutcome(
                f"订单 {state.order_id} 包含：{choices}。请明确需要处理哪一件商品。",
                state.stage,
            )

        current_scope = sorted(set(resolved) - set(excluded))
        if not current_scope:
            state.missing_fields = ["target_item"]
            transition_state(state, TaskStage.COLLECTING_INFO)
            return PolicyOutcome("你排除了所有商品，请重新说明需要退哪一件。", state.stage)
        if current_scope != state.selected_item_ids or sorted(excluded) != state.excluded_item_ids:
            if state.pending_action:
                state.pending_action = None
                state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            state.selected_item_ids = current_scope
            state.excluded_item_ids = sorted(excluded)
            state.slots.pop("refund_amount", None)

        if state.stage == TaskStage.AWAITING_CONFIRMATION:
            transition_state(state, TaskStage.CHECKING_POLICY)
        else:
            transition_state(state, TaskStage.CHECKING_POLICY)
        evidence: list[str] = []
        for item_id in state.selected_item_ids:
            result = self._observe(
                state,
                "search_policy",
                {"order_id": state.order_id, "item_id": item_id, "reason": state.reason},
            )
            if not result.ok:
                return self._tool_failure(state, result, "政策检索失败")
            evidence.extend(doc["id"] for doc in result.data.get("evidence", []))
        state.policy_evidence = sorted(set(evidence))

        calculation = self._observe(
            state,
            "calculate_refund",
            {
                "order_id": state.order_id,
                "item_ids": state.selected_item_ids,
                "reason": state.reason,
            },
        )
        if not calculation.ok:
            return self._tool_failure(state, calculation, "退款计算失败")
        ineligible = calculation.data.get("ineligible_items", [])
        if ineligible:
            state.pending_action = None
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            transition_state(state, TaskStage.REJECTED)
            reasons = "；".join(item["reason"] for item in ineligible)
            return PolicyOutcome(f"当前不能自动退货：{reasons}", state.stage)

        amount = float(calculation.data["refund_amount"])
        state.set_slot(
            "refund_amount",
            amount,
            SlotSource.TOOL,
            1.0,
            "calculate_refund",
        )
        state.missing_fields = []
        item_names = [
            item["name"] for item in order.get("items", []) if item["item_id"] in current_scope
        ]

        if state.consultation_only:
            state.pending_action = None
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            state.risk_level = RiskLevel.MEDIUM
            transition_state(state, TaskStage.COLLECTING_INFO)
            return PolicyOutcome(
                f"经核验，{'、'.join(item_names)}预计可退 {amount:.2f} 元。"
                "这次只做咨询，没有创建退货申请。",
                state.stage,
            )

        if state.confirmation_status == ConfirmationStatus.CONFIRMED:
            pending = state.pending_action
            fresh_scope = (state.order_id, tuple(sorted(current_scope)), round(amount, 2), state.reason)
            if pending is None or pending.scope_key() != fresh_scope:
                state.pending_action = None
                state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
                state.conflicts.append("CONFIRMATION_SCOPE_MISMATCH")
            else:
                transition_state(state, TaskStage.EXECUTING)
                state.risk_level = RiskLevel.HIGH
                creation = self._observe(
                    state,
                    "create_return_request",
                    {
                        "order_id": pending.order_id,
                        "item_ids": pending.item_ids,
                        "refund_amount": pending.amount,
                        "reason": pending.reason,
                        "idempotency_key": pending.idempotency_key,
                        "user_confirmation": True,
                    },
                )
                if creation.status == "UNKNOWN_COMMIT":
                    reconciliation = self._observe(
                        state,
                        "get_return_status",
                        {"idempotency_key": pending.idempotency_key},
                    )
                    request = reconciliation.data.get("request") if reconciliation.ok else None
                    if request:
                        state.pending_action = None
                        state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
                        transition_state(state, TaskStage.COMPLETED)
                        return PolicyOutcome(
                            f"写入响应丢失，但对账确认退货申请 {request['request_id']} 已创建。",
                            state.stage,
                        )
                    return self._handoff_with_reason(state, "退货操作提交状态未知")
                if not creation.ok:
                    return self._tool_failure(state, creation, "创建退货申请失败")
                request = creation.data
                state.pending_action = None
                state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
                transition_state(state, TaskStage.COMPLETED)
                return PolicyOutcome(
                    f"退货申请 {request['request_id']} 已创建：{'、'.join(item_names)}，"
                    f"退款金额 {request['refund_amount']:.2f} 元。",
                    state.stage,
                )

        pending = PendingAction(
            action="create_return_request",
            order_id=state.order_id,
            item_ids=list(current_scope),
            amount=amount,
            reason=state.reason,
            idempotency_key=(
                f"structured:{state.task_id}:{state.order_id}:"
                f"{','.join(sorted(current_scope))}:{amount:.2f}:{state.reason}"
            ),
            requested_turn=state.turn_index,
        )
        state.pending_action = pending
        state.confirmation_status = ConfirmationStatus.PENDING
        state.risk_level = RiskLevel.HIGH
        transition_state(state, TaskStage.AWAITING_CONFIRMATION)
        excluded_text = (
            "；已排除其他商品" if state.excluded_item_ids else ""
        )
        return PolicyOutcome(
            f"将为{'、'.join(item_names)}申请退货，预计退款 {amount:.2f} 元{excluded_text}。"
            "如需执行，请回复“确认退货”。",
            state.stage,
        )

    def _resolve_items(self, state: StructuredTaskState, items: list[dict]) -> tuple[list[str], list[str], list[str]]:
        valid_ids = {item["item_id"] for item in items}
        selected = [item_id for item_id in state.selected_item_ids if item_id in valid_ids]
        excluded = [item_id for item_id in state.excluded_item_ids if item_id in valid_ids]
        target_slot = state.slots.get("target_product_mentions")
        excluded_slot = state.slots.get("excluded_product_mentions")
        unresolved: list[str] = []
        for mention in target_slot.value if target_slot else []:
            matches = [item["item_id"] for item in items if self._name_matches(mention, item["name"])]
            if len(matches) == 1:
                selected.extend(matches)
            else:
                unresolved.append(mention)
        for mention in excluded_slot.value if excluded_slot else []:
            matches = [item["item_id"] for item in items if self._name_matches(mention, item["name"])]
            if len(matches) == 1:
                excluded.extend(matches)
            else:
                unresolved.append(mention)
        return sorted(set(selected)), sorted(set(excluded)), sorted(set(unresolved))

    @staticmethod
    def _name_matches(mention: str, item_name: str) -> bool:
        normalized = mention.replace(" ", "")
        name = item_name.replace(" ", "")
        if normalized in name or name in normalized:
            return True
        aliases = {
            "鞋": "运动鞋",
            "鞋子": "运动鞋",
            "耳机": "蓝牙耳机",
            "杯": "定制马克杯",
            "杯子": "定制马克杯",
            "马克杯": "定制马克杯",
            "键盘": "机械键盘",
        }
        return aliases.get(normalized) == name

    def _observe(self, state: StructuredTaskState, tool_name: str, arguments: dict) -> ToolResult:
        result = self.call_tool(tool_name, arguments)
        state.facts[f"{tool_name}:{len(state.facts) + 1}"] = ToolFact(
            key=tool_name,
            value=result.data if result.ok else {"error_code": result.error_code},
            tool=tool_name,
            observed_turn=state.turn_index,
        )
        return result

    def _handoff(self, state: StructuredTaskState) -> PolicyOutcome:
        return self._handoff_with_reason(state, "用户要求人工处理")

    def _handoff_with_reason(self, state: StructuredTaskState, reason: str) -> PolicyOutcome:
        result = self._observe(
            state,
            "create_ticket",
            {
                "subject": "售后人工处理",
                "description": reason,
                **({"order_id": state.order_id} if state.order_id else {}),
            },
        )
        transition_state(state, TaskStage.HANDOFF)
        if result.ok:
            ticket = result.data["ticket"]
            return PolicyOutcome(
                f"已创建人工工单 {ticket['ticket_id']}，请等待客服处理。", state.stage
            )
        return PolicyOutcome(f"暂时无法创建人工工单：{result.message}", state.stage)

    def _tool_failure(
        self, state: StructuredTaskState, result: ToolResult, prefix: str
    ) -> PolicyOutcome:
        if state.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL} or not result.retryable:
            return self._handoff_with_reason(state, f"{prefix}：{result.error_code or result.status}")
        transition_state(state, TaskStage.COLLECTING_INFO)
        return PolicyOutcome(f"{prefix}：{result.message}，请稍后重试。", state.stage)
