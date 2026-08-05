"""Deterministic in-memory business backend used by the MVP and scenarios."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

from domain.models import (
    OrderStatus,
    RefundMethod,
    RefundStatus,
    RefundTransaction,
    ReturnItemStatus,
    ReturnRequest,
    ReturnRequestStatus,
    ShipmentStatus,
    StatusChange,
    Ticket,
    TicketPriority,
    TicketStatus,
    TicketType,
    User,
)
from domain.state_machine import (
    InvalidTransition,
    ORDER_STATE_MACHINE,
    REFUND_STATE_MACHINE,
    RETURN_STATE_MACHINE,
    SHIPMENT_STATE_MACHINE,
    TICKET_STATE_MACHINE,
)
from seed.loader import load_seed_data


DEFAULT_BUSINESS_DATE = date(2026, 7, 16)


class ToolError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class InMemoryStore:
    """Business-facing repository with explicit records and guarded writes."""

    def __init__(self, seed_dir: Path | None = None, today: date = DEFAULT_BUSINESS_DATE) -> None:
        data = load_seed_data(seed_dir)
        self.today = today
        self.users: dict[str, User] = data["users"]
        self.products = data["products"]
        self.orders = data["orders"]
        self.shipments = data["shipments"]
        self.return_requests: dict[str, ReturnRequest] = {}
        self.refund_transactions: dict[str, RefundTransaction] = {}
        self.tickets: dict[str, Ticket] = {}
        self.idempotency_index: dict[str, str] = {}

    def get_user(self, user_id: str) -> User:
        user = self.users.get(user_id)
        if user is None:
            raise ToolError("USER_NOT_FOUND", f"用户 {user_id} 不存在。")
        if str(user.status) != "active":
            raise ToolError("USER_INACTIVE", "当前用户账户不可进行售后操作。")
        return user

    def get_product(self, product_id: str):
        product = self.products.get(product_id)
        if product is None:
            raise ToolError("PRODUCT_NOT_FOUND", f"商品 {product_id} 不存在。")
        return product

    def get_order(self, user_id: str, order_id: str):
        self.get_user(user_id)
        order = self.orders.get(order_id)
        if order is None:
            raise ToolError("ORDER_NOT_FOUND", f"订单 {order_id} 不存在。")
        if order.user_id != user_id:
            raise ToolError("FORBIDDEN", "当前用户无权访问该订单。")
        return order

    def list_orders(self, user_id: str):
        self.get_user(user_id)
        return [order for order in self.orders.values() if order.user_id == user_id]

    def get_shipment(self, user_id: str, shipment_id: str):
        shipment = self.shipments.get(shipment_id)
        if shipment is None:
            raise ToolError("SHIPMENT_NOT_FOUND", f"物流单 {shipment_id} 不存在。")
        self.get_order(user_id, shipment.order_id)
        return shipment

    def list_shipments(self, user_id: str, order_id: str):
        self.get_order(user_id, order_id)
        return [shipment for shipment in self.shipments.values() if shipment.order_id == order_id]

    def transition_order(self, order_id: str, target: OrderStatus, actor: str, reason: str = ""):
        order = self.orders[order_id]
        source = str(order.status)
        ORDER_STATE_MACHINE.transition(source, target.value)
        if source != target.value:
            now = datetime.now()
            order.status_history.append(StatusChange(source, target.value, now, actor, reason))
            order.status = target
        return order

    def transition_shipment(
        self, shipment_id: str, target: ShipmentStatus, actor: str, reason: str = ""
    ):
        shipment = self.shipments[shipment_id]
        source = str(shipment.status)
        SHIPMENT_STATE_MACHINE.transition(source, target.value)
        if source != target.value:
            now = datetime.now()
            shipment.status_history.append(StatusChange(source, target.value, now, actor, reason))
            shipment.status = target
        return shipment

    def check_eligibility(self, item, reason: str) -> tuple[bool, str]:
        if item.return_status != ReturnItemStatus.NOT_REQUESTED:
            return False, f"{item.name} 已有售后申请，不能重复创建。"
        if item.delivered_on is None:
            return False, f"{item.name} 仍在运输中，签收后才能申请退货。"
        elapsed_days = (self.today - item.delivered_on).days
        if item.is_custom and reason != "quality_issue":
            return False, f"{item.name} 是定制商品，不支持无理由退货。"
        allowed_days = 30 if reason == "quality_issue" else 7
        if elapsed_days > allowed_days:
            return False, f"{item.name} 已签收 {elapsed_days} 天，超过 {allowed_days} 天售后期限。"
        return True, "eligible"

    def create_return_request(
        self,
        user_id: str,
        order_id: str,
        item_ids: list[str],
        refund_amount: float,
        reason: str,
        idempotency_key: str,
        user_confirmation: bool,
    ) -> ReturnRequest:
        """Create a return record and a separate pending refund transaction."""

        if not user_confirmation:
            raise ToolError("CONFIRMATION_REQUIRED", "高风险操作需要用户明确确认。")
        if idempotency_key in self.idempotency_index:
            request_id = self.idempotency_index[idempotency_key]
            return self.return_requests[request_id]

        order = self.get_order(user_id, order_id)
        if not item_ids:
            raise ToolError("EMPTY_ITEMS", "必须选择至少一个商品。")
        items_by_id = {item.item_id: item for item in order.items}
        if any(item_id not in items_by_id for item_id in item_ids):
            raise ToolError("ITEM_NOT_IN_ORDER", "存在不属于该订单的商品。")

        actual_amount = 0.0
        for item_id in item_ids:
            item = items_by_id[item_id]
            eligible, message = self.check_eligibility(item, reason)
            if not eligible:
                raise ToolError("POLICY_REJECTED", message)
            actual_amount += item.price * item.quantity

        if round(actual_amount, 2) != round(refund_amount, 2):
            raise ToolError(
                "AMOUNT_MISMATCH",
                f"退款金额校验失败：服务端计算为 {actual_amount:.2f} 元。",
            )

        now = datetime.now()
        request_id = f"R{len(self.return_requests) + 1:05d}"
        request = ReturnRequest(
            request_id=request_id,
            user_id=user_id,
            order_id=order_id,
            item_ids=sorted(item_ids),
            refund_amount=round(actual_amount, 2),
            reason=reason,
            status=ReturnRequestStatus.SUBMITTED,
            idempotency_key=idempotency_key,
            created_at=now,
            timeline=[
                StatusChange(
                    from_status=None,
                    to_status=ReturnRequestStatus.SUBMITTED.value,
                    changed_at=now,
                    actor="customer_service_agent",
                    reason="用户确认后提交退货申请",
                )
            ],
        )
        self.return_requests[request_id] = request
        self.idempotency_index[idempotency_key] = request_id

        refund_id = f"RF{len(self.refund_transactions) + 1:05d}"
        self.refund_transactions[refund_id] = RefundTransaction(
            refund_id=refund_id,
            user_id=user_id,
            order_id=order_id,
            request_id=request_id,
            item_ids=sorted(item_ids),
            amount=round(actual_amount, 2),
            method=RefundMethod.ORIGINAL_PAYMENT,
            status=RefundStatus.PENDING,
            created_at=now,
            timeline=[
                StatusChange(
                    from_status=None,
                    to_status=RefundStatus.PENDING.value,
                    changed_at=now,
                    actor="refund_service",
                    reason="退货申请创建，等待退款处理",
                )
            ],
        )
        for item_id in item_ids:
            items_by_id[item_id].return_status = ReturnItemStatus.REQUESTED
        return request

    def transition_return(self, request_id: str, target: ReturnRequestStatus, actor: str, reason: str = ""):
        request = self.return_requests[request_id]
        source = str(request.status)
        RETURN_STATE_MACHINE.transition(source, target.value)
        if source != target.value:
            now = datetime.now()
            request.timeline.append(StatusChange(source, target.value, now, actor, reason))
            request.status = target
        return request

    def transition_refund(self, refund_id: str, target: RefundStatus, actor: str, reason: str = ""):
        refund = self.refund_transactions[refund_id]
        source = str(refund.status)
        REFUND_STATE_MACHINE.transition(source, target.value)
        if source != target.value:
            now = datetime.now()
            refund.timeline.append(StatusChange(source, target.value, now, actor, reason))
            refund.status = target
            if target == RefundStatus.SUCCEEDED:
                refund.settled_at = now
        return refund

    def create_ticket(
        self,
        user_id: str,
        subject: str,
        description: str,
        ticket_type: TicketType = TicketType.AFTER_SALES,
        priority: TicketPriority = TicketPriority.NORMAL,
        order_id: str | None = None,
    ) -> Ticket:
        if order_id is not None:
            self.get_order(user_id, order_id)
        now = datetime.now()
        ticket_id = f"T{len(self.tickets) + 1:05d}"
        ticket = Ticket(
            ticket_id=ticket_id,
            user_id=user_id,
            order_id=order_id,
            ticket_type=ticket_type,
            priority=priority,
            subject=subject,
            description=description,
            status=TicketStatus.OPEN,
            assignee=None,
            created_at=now,
            updated_at=now,
            timeline=[StatusChange(None, TicketStatus.OPEN.value, now, "system", "创建工单")],
        )
        self.tickets[ticket_id] = ticket
        return ticket

    def transition_ticket(self, ticket_id: str, target: TicketStatus, actor: str, reason: str = ""):
        ticket = self.tickets[ticket_id]
        source = str(ticket.status)
        TICKET_STATE_MACHINE.transition(source, target.value)
        if source != target.value:
            now = datetime.now()
            ticket.timeline.append(StatusChange(source, target.value, now, actor, reason))
            ticket.status = target
            ticket.updated_at = now
        return ticket

    def get_return_by_idempotency(self, idempotency_key: str) -> Optional[ReturnRequest]:
        request_id = self.idempotency_index.get(idempotency_key)
        return self.return_requests.get(request_id) if request_id else None
