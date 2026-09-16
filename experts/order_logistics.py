"""Order/logistics expert: read operations and order-item grounding only."""

from __future__ import annotations

from routing.models import ExpertName, ExpertResult, ExpertStatus
from structured.models import ConfirmationStatus, Intent, StructuredTaskState, TaskStage
from structured.state_machine import transition_state

from .base import BaseExpert


class OrderLogisticsExpert(BaseExpert):
    name = ExpertName.ORDER_LOGISTICS
    allowed_tools = frozenset({"list_orders", "get_order", "list_shipments"})

    def execute(self, state: StructuredTaskState) -> ExpertResult:
        if state.intent in {Intent.QUERY_ORDER, Intent.QUERY_LOGISTICS}:
            return self._query(state)
        if state.intent == Intent.REQUEST_RETURN:
            return self._ground_return_scope(state)
        return ExpertResult(
            expert=self.name,
            status=ExpertStatus.ESCALATE,
            reason_code="UNSUPPORTED_INTENT",
            escalation_reason="订单专家收到不支持的任务",
        )

    def _query(self, state: StructuredTaskState) -> ExpertResult:
        if not state.order_id:
            result = self.observe(state, "list_orders", {})
            if not result.ok:
                return self.escalate("订单列表查询失败", result)
            order_ids = [order["order_id"] for order in result.data.get("orders", [])]
            state.missing_fields = ["order_id"]
            transition_state(state, TaskStage.COLLECTING_INFO)
            choices = "、".join(order_ids) if order_ids else "没有可用订单"
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.CLARIFY,
                response=f"我找到了这些订单：{choices}。请明确告诉我需要查询哪一单。",
                reason_code="ORDER_ID_REQUIRED",
            )

        transition_state(state, TaskStage.RESOLVING_ENTITIES)
        order_result = self.observe(state, "get_order", {"order_id": state.order_id})
        if not order_result.ok:
            return self.escalate("订单查询失败", order_result)
        order = order_result.data["order"]
        transition_state(state, TaskStage.EXECUTING)
        if state.intent == Intent.QUERY_LOGISTICS:
            shipment_result = self.observe(
                state, "list_shipments", {"order_id": state.order_id}
            )
            if not shipment_result.ok:
                return self.escalate("物流查询失败", shipment_result)
            shipments = shipment_result.data.get("shipments", [])
            parts = [
                f"{shipment['carrier']} {shipment['tracking_number']}：{shipment['status']}"
                for shipment in shipments
            ]
            response = (
                f"订单 {state.order_id} 有 {len(shipments)} 个包裹："
                + "；".join(parts)
                + "。"
            )
        else:
            items = "、".join(
                f"{item['name']}（{item['item_id']}）" for item in order.get("items", [])
            )
            response = f"订单 {state.order_id} 状态为 {order['status']}，包含：{items}。"
        state.missing_fields = []
        transition_state(state, TaskStage.COMPLETED)
        return ExpertResult(
            expert=self.name,
            status=ExpertStatus.COMPLETE,
            response=response,
            reason_code="QUERY_COMPLETED",
        )

    def _ground_return_scope(self, state: StructuredTaskState) -> ExpertResult:
        if not state.order_id:
            state.missing_fields = ["order_id"]
            transition_state(state, TaskStage.COLLECTING_INFO)
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.CLARIFY,
                response="为避免处理错订单，请先提供订单号。",
                reason_code="ORDER_ID_REQUIRED",
            )

        if state.stage != TaskStage.AWAITING_CONFIRMATION:
            transition_state(state, TaskStage.RESOLVING_ENTITIES)
        order_result = self.observe(state, "get_order", {"order_id": state.order_id})
        if not order_result.ok:
            return self.escalate("订单查询失败", order_result)
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
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.CLARIFY,
                response=f"订单 {state.order_id} 包含：{choices}。请明确需要处理哪一件商品。",
                reason_code="TARGET_ITEM_REQUIRED",
            )

        current_scope = sorted(set(resolved) - set(excluded))
        if not current_scope:
            state.missing_fields = ["target_item"]
            transition_state(state, TaskStage.COLLECTING_INFO)
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.CLARIFY,
                response="你排除了所有商品，请重新说明需要退哪一件。",
                reason_code="EMPTY_ITEM_SCOPE",
            )
        normalized_excluded = sorted(excluded)
        if current_scope != state.selected_item_ids or normalized_excluded != state.excluded_item_ids:
            if state.pending_action:
                state.pending_action = None
                state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            state.selected_item_ids = current_scope
            state.excluded_item_ids = normalized_excluded
            state.slots.pop("refund_amount", None)
            state.policy_evidence = []
        state.missing_fields = []
        item_names = {
            item["item_id"]: item["name"]
            for item in order.get("items", [])
            if item["item_id"] in current_scope
        }
        return ExpertResult(
            expert=self.name,
            status=ExpertStatus.CONTINUE,
            next_expert=ExpertName.POLICY,
            reason_code="RETURN_SCOPE_GROUNDED",
            facts={"item_names": item_names},
        )

    def _resolve_items(
        self, state: StructuredTaskState, items: list[dict]
    ) -> tuple[list[str], list[str], list[str]]:
        valid_ids = {item["item_id"] for item in items}
        selected = [item_id for item_id in state.selected_item_ids if item_id in valid_ids]
        excluded = [item_id for item_id in state.excluded_item_ids if item_id in valid_ids]
        target_slot = state.slots.get("target_product_mentions")
        excluded_slot = state.slots.get("excluded_product_mentions")
        unresolved: list[str] = []
        for mention in target_slot.value if target_slot else []:
            matches = [
                item["item_id"] for item in items if self._name_matches(mention, item["name"])
            ]
            if len(matches) == 1:
                selected.extend(matches)
            else:
                unresolved.append(mention)
        for mention in excluded_slot.value if excluded_slot else []:
            matches = [
                item["item_id"] for item in items if self._name_matches(mention, item["name"])
            ]
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
