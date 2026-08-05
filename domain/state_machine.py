"""Formal state transition tables for business aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


class InvalidTransition(ValueError):
    """Raised when a business record attempts an illegal state transition."""


@dataclass(frozen=True)
class TransitionRule:
    source: str
    target: str


class StateMachine:
    def __init__(self, name: str, transitions: Iterable[TransitionRule]) -> None:
        self.name = name
        self._transitions = {(rule.source, rule.target) for rule in transitions}

    def can_transition(self, source: str, target: str) -> bool:
        return source == target or (source, target) in self._transitions

    def transition(self, source: str, target: str) -> str:
        if not self.can_transition(source, target):
            raise InvalidTransition(
                f"{self.name} 不允许从 {source} 转换到 {target}。"
            )
        return target

    @property
    def transitions(self) -> frozenset[tuple[str, str]]:
        """Expose the transition table for documentation and tests."""

        return frozenset(self._transitions)


ORDER_STATE_MACHINE = StateMachine(
    "order",
    [
        TransitionRule("created", "paid"),
        TransitionRule("paid", "partially_shipped"),
        TransitionRule("paid", "shipped"),
        TransitionRule("partially_shipped", "partially_delivered"),
        TransitionRule("partially_shipped", "shipped"),
        TransitionRule("shipped", "delivered"),
        TransitionRule("partially_delivered", "delivered"),
        TransitionRule("delivered", "completed"),
        TransitionRule("created", "cancelled"),
        TransitionRule("paid", "cancelled"),
    ],
)

SHIPMENT_STATE_MACHINE = StateMachine(
    "shipment",
    [
        TransitionRule("created", "picked_up"),
        TransitionRule("picked_up", "in_transit"),
        TransitionRule("in_transit", "out_for_delivery"),
        TransitionRule("out_for_delivery", "delivered"),
        TransitionRule("in_transit", "exception"),
        TransitionRule("exception", "in_transit"),
    ],
)

RETURN_STATE_MACHINE = StateMachine(
    "return_request",
    [
        TransitionRule("draft", "submitted"),
        TransitionRule("submitted", "approved"),
        TransitionRule("submitted", "rejected"),
        TransitionRule("approved", "returning"),
        TransitionRule("returning", "received"),
        TransitionRule("received", "completed"),
        TransitionRule("draft", "cancelled"),
        TransitionRule("submitted", "cancelled"),
        TransitionRule("approved", "cancelled"),
    ],
)

REFUND_STATE_MACHINE = StateMachine(
    "refund_transaction",
    [
        TransitionRule("pending", "processing"),
        TransitionRule("processing", "succeeded"),
        TransitionRule("processing", "failed"),
        TransitionRule("processing", "unknown"),
        TransitionRule("unknown", "processing"),
    ],
)

TICKET_STATE_MACHINE = StateMachine(
    "ticket",
    [
        TransitionRule("open", "assigned"),
        TransitionRule("open", "in_progress"),
        TransitionRule("assigned", "in_progress"),
        TransitionRule("in_progress", "waiting_for_user"),
        TransitionRule("waiting_for_user", "in_progress"),
        TransitionRule("in_progress", "resolved"),
        TransitionRule("resolved", "closed"),
        TransitionRule("resolved", "in_progress"),
    ],
)
