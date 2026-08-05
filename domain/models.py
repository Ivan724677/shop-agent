"""Explicit domain models used by the business simulator.

The models intentionally keep business records separate from agent state.  In
particular, a product is a catalog record while an OrderItem is a purchase-time
snapshot, and a ReturnRequest is separate from a RefundTransaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class ValueEnum(str, Enum):
    """String enum that can be stored in JSON and compared with string values."""

    def __str__(self) -> str:
        return self.value


class UserStatus(ValueEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class MembershipLevel(ValueEnum):
    STANDARD = "standard"
    SILVER = "silver"
    GOLD = "gold"


class OrderStatus(ValueEnum):
    CREATED = "created"
    PAID = "paid"
    PARTIALLY_SHIPPED = "partially_shipped"
    SHIPPED = "shipped"
    PARTIALLY_DELIVERED = "partially_delivered"
    DELIVERED = "delivered"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ReturnItemStatus(ValueEnum):
    NOT_REQUESTED = "not_requested"
    REQUESTED = "requested"
    RETURNING = "returning"
    RETURNED = "returned"


class ShipmentStatus(ValueEnum):
    CREATED = "created"
    PICKED_UP = "picked_up"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    EXCEPTION = "exception"


class ReturnRequestStatus(ValueEnum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    RETURNING = "returning"
    RECEIVED = "received"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RefundStatus(ValueEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RefundMethod(ValueEnum):
    ORIGINAL_PAYMENT = "original_payment"
    STORE_CREDIT = "store_credit"


class TicketStatus(ValueEnum):
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_USER = "waiting_for_user"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TicketPriority(ValueEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TicketType(ValueEnum):
    AFTER_SALES = "after_sales"
    LOGISTICS = "logistics"
    PAYMENT = "payment"
    COMPLAINT = "complaint"


@dataclass
class DictLike:
    """Small compatibility helper for existing demos that index records by key."""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class Address(DictLike):
    address_id: str
    recipient: str
    phone: str
    province: str
    city: str
    district: str
    detail: str
    is_default: bool = False


@dataclass
class User(DictLike):
    user_id: str
    name: str
    phone: str
    email: str
    addresses: list[Address] = field(default_factory=list)
    membership_level: MembershipLevel = MembershipLevel.STANDARD
    status: UserStatus = UserStatus.ACTIVE

    @property
    def default_address(self) -> Address | None:
        return next((address for address in self.addresses if address.is_default), None)


@dataclass
class Product(DictLike):
    product_id: str
    name: str
    category: str
    current_price: float
    is_custom: bool = False
    return_tags: set[str] = field(default_factory=set)
    active: bool = True


@dataclass
class OrderItem(DictLike):
    item_id: str
    product_id: str
    name: str
    price: float
    category: str
    delivered_on: date | None
    quantity: int = 1
    is_custom: bool = False
    return_status: ReturnItemStatus = ReturnItemStatus.NOT_REQUESTED
    shipment_id: str | None = None


@dataclass
class Order(DictLike):
    order_id: str
    user_id: str
    status: OrderStatus
    items: list[OrderItem]
    created_at: datetime | None = None
    payment_method: str = "alipay"
    status_history: list["StatusChange"] = field(default_factory=list)


@dataclass
class TrackingEvent(DictLike):
    event_id: str
    status: ShipmentStatus
    occurred_at: datetime
    location: str
    description: str


@dataclass
class Shipment(DictLike):
    shipment_id: str
    order_id: str
    tracking_number: str
    carrier: str
    status: ShipmentStatus
    item_ids: list[str]
    events: list[TrackingEvent] = field(default_factory=list)
    shipped_at: datetime | None = None
    estimated_delivery: date | None = None
    delivered_on: date | None = None
    status_history: list["StatusChange"] = field(default_factory=list)


@dataclass
class StatusChange(DictLike):
    from_status: str | None
    to_status: str
    changed_at: datetime
    actor: str
    reason: str = ""


@dataclass
class ReturnRequest(DictLike):
    request_id: str
    user_id: str
    order_id: str
    item_ids: list[str]
    refund_amount: float
    reason: str
    status: ReturnRequestStatus
    idempotency_key: str
    created_at: datetime
    timeline: list[StatusChange] = field(default_factory=list)


@dataclass
class RefundTransaction(DictLike):
    refund_id: str
    user_id: str
    order_id: str
    request_id: str
    item_ids: list[str]
    amount: float
    method: RefundMethod
    status: RefundStatus
    created_at: datetime
    settled_at: datetime | None = None
    timeline: list[StatusChange] = field(default_factory=list)


@dataclass
class Ticket(DictLike):
    ticket_id: str
    user_id: str
    order_id: str | None
    ticket_type: TicketType
    priority: TicketPriority
    subject: str
    description: str
    status: TicketStatus
    assignee: str | None
    created_at: datetime
    updated_at: datetime
    timeline: list[StatusChange] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
