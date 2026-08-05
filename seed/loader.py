"""Load deterministic JSON fixtures into domain objects."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from domain.models import (
    Address,
    MembershipLevel,
    Order,
    OrderItem,
    OrderStatus,
    Product,
    ReturnItemStatus,
    Shipment,
    ShipmentStatus,
    TrackingEvent,
    User,
    UserStatus,
)


DEFAULT_SEED_DIR = Path(__file__).parent / "data"


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _read(seed_dir: Path, name: str) -> list[dict[str, Any]]:
    with (seed_dir / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def load_seed_data(seed_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Return all seed data as typed domain records keyed by their IDs."""

    seed_dir = seed_dir or DEFAULT_SEED_DIR

    users: dict[str, User] = {}
    for raw in _read(seed_dir, "users.json"):
        addresses = [Address(**address) for address in raw.pop("addresses", [])]
        membership_level = MembershipLevel(raw.pop("membership_level"))
        status = UserStatus(raw.pop("status"))
        users[raw["user_id"]] = User(
            **raw,
            addresses=addresses,
            membership_level=membership_level,
            status=status,
        )

    products: dict[str, Product] = {}
    for raw in _read(seed_dir, "products.json"):
        return_tags = set(raw.pop("return_tags"))
        products[raw["product_id"]] = Product(**raw, return_tags=return_tags)

    orders: dict[str, Order] = {}
    for raw in _read(seed_dir, "orders.json"):
        items = []
        for item in raw.pop("items", []):
            delivered_on = _date(item.pop("delivered_on"))
            return_status = ReturnItemStatus(item.pop("return_status"))
            items.append(
                OrderItem(
                    **item,
                    delivered_on=delivered_on,
                    return_status=return_status,
                )
            )
        status = OrderStatus(raw.pop("status"))
        created_at = _datetime(raw.pop("created_at"))
        orders[raw["order_id"]] = Order(
            **raw,
            status=status,
            created_at=created_at,
            items=items,
        )

    shipments: dict[str, Shipment] = {}
    for raw in _read(seed_dir, "shipments.json"):
        events = []
        for event in raw.pop("events", []):
            event_status = ShipmentStatus(event.pop("status"))
            occurred_at = datetime.fromisoformat(event.pop("occurred_at"))
            events.append(TrackingEvent(**event, status=event_status, occurred_at=occurred_at))
        status = ShipmentStatus(raw.pop("status"))
        shipped_at = _datetime(raw.pop("shipped_at"))
        estimated_delivery = _date(raw.pop("estimated_delivery"))
        delivered_on = _date(raw.pop("delivered_on"))
        shipments[raw["shipment_id"]] = Shipment(
            **raw,
            status=status,
            shipped_at=shipped_at,
            estimated_delivery=estimated_delivery,
            delivered_on=delivered_on,
            events=events,
        )

    return {
        "users": users,
        "products": products,
        "orders": orders,
        "shipments": shipments,
    }
