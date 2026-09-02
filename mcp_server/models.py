"""Protocol, authorization and audit models for the MCP reliability layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    session_id: str
    scopes: frozenset[str]
    actor_type: str = "customer"


@dataclass
class AuditEvent:
    audit_id: str
    timestamp: str
    correlation_id: str
    request_id: str
    actor_id: str
    actor_type: str
    tool: str
    risk_level: str
    side_effect: bool
    decision: str
    status: str
    error_code: str | None
    attempts: int
    duration_ms: float
    circuit_before: str
    circuit_after: str
    argument_digest: str
    sanitized_arguments: dict[str, Any] = field(default_factory=dict)
    idempotency_key_digest: str | None = None

