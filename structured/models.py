"""阶段三结构化 Agent 的状态模型定义（纯数据结构，不含执行逻辑）。

- SemanticFrame：LLM 语义解析产出的候选帧，无执行权限，需经 reducer 定权
- SlotValue：带来源/置信度/证据/更新轮次的槽位值（可审计）
- PendingAction：冻结的待确认动作，scope_key() 绑定订单/商品/金额/原因
- StructuredTaskState：工作记忆（当前任务的槽位/阶段/风险/确认状态）
- TurnMemory：情节记忆（每轮输入、状态前后快照、工具调用与回复）

所有枚举（Intent/TaskStage/RiskLevel/ConfirmationStatus/SlotSource）
继承 StringEnum，可 JSON 序列化并与字符串直接比较。
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Intent(StringEnum):
    UNKNOWN = "unknown"
    KEEP_CURRENT = "keep_current"
    QUERY_ORDER = "query_order"
    QUERY_LOGISTICS = "query_logistics"
    REQUEST_RETURN = "request_return"
    HUMAN_HANDOFF = "human_handoff"


class TaskStage(StringEnum):
    IDLE = "idle"
    COLLECTING_INFO = "collecting_info"
    RESOLVING_ENTITIES = "resolving_entities"
    CHECKING_POLICY = "checking_policy"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    HANDOFF = "handoff"


class RiskLevel(StringEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfirmationStatus(StringEnum):
    NOT_REQUESTED = "not_requested"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class SlotSource(StringEnum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"
    TOOL = "tool"
    DERIVED = "derived"
    USER_CONFIRMED = "user_confirmed"


@dataclass
class SlotValue:
    value: Any
    source: SlotSource
    confidence: float
    evidence: str
    updated_turn: int


@dataclass
class SemanticFrame:
    """LLM-generated candidate interpretation; never directly authorizes actions."""

    intent: Intent = Intent.UNKNOWN
    order_id: str | None = None
    item_ids: list[str] = field(default_factory=list)
    product_mentions: list[str] = field(default_factory=list)
    excluded_item_ids: list[str] = field(default_factory=list)
    excluded_product_mentions: list[str] = field(default_factory=list)
    reason: str | None = None
    consultation_only: bool = False
    confirmation: str = "none"
    wants_handoff: bool = False
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SemanticFrame":
        allowed_intents = {intent.value for intent in Intent}
        intent_value = raw.get("intent", Intent.UNKNOWN.value)
        intent = Intent(intent_value) if intent_value in allowed_intents else Intent.UNKNOWN
        confirmation = raw.get("confirmation", "none")
        if confirmation not in {"none", "confirm", "reject"}:
            confirmation = "none"
        reason = raw.get("reason")
        if reason not in {None, "no_reason_return", "quality_issue"}:
            reason = None
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            intent=intent,
            order_id=raw.get("order_id") if isinstance(raw.get("order_id"), str) else None,
            item_ids=_string_list(raw.get("item_ids")),
            product_mentions=_string_list(raw.get("product_mentions")),
            excluded_item_ids=_string_list(raw.get("excluded_item_ids")),
            excluded_product_mentions=_string_list(raw.get("excluded_product_mentions")),
            reason=reason,
            consultation_only=bool(raw.get("consultation_only", False)),
            confirmation=confirmation,
            wants_handoff=bool(raw.get("wants_handoff", False)),
            confidence=max(0.0, min(confidence, 1.0)),
        )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


@dataclass
class DeterministicSignals:
    order_id: str | None = None
    item_ids: list[str] = field(default_factory=list)
    product_mentions: list[str] = field(default_factory=list)
    excluded_item_ids: list[str] = field(default_factory=list)
    excluded_product_mentions: list[str] = field(default_factory=list)
    explicit_intent: Intent = Intent.UNKNOWN
    reason: str | None = None
    consultation_only: bool = False
    confirmation: str = "none"
    relative_order_reference: bool = False


@dataclass
class PendingAction:
    action: str
    order_id: str
    item_ids: list[str]
    amount: float
    reason: str
    idempotency_key: str
    requested_turn: int

    def scope_key(self) -> tuple[str, tuple[str, ...], float, str]:
        return (
            self.order_id,
            tuple(sorted(self.item_ids)),
            round(self.amount, 2),
            self.reason,
        )


@dataclass
class ToolFact:
    key: str
    value: Any
    tool: str
    observed_turn: int


@dataclass
class TurnMemory:
    turn: int
    user_message: str
    semantic_frame: dict[str, Any]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    tool_names: list[str]
    response: str


@dataclass
class StructuredTaskState:
    """Working memory used by deterministic dialogue policy."""

    user_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_index: int = 0
    intent: Intent = Intent.UNKNOWN
    stage: TaskStage = TaskStage.IDLE
    risk_level: RiskLevel = RiskLevel.LOW
    slots: dict[str, SlotValue] = field(default_factory=dict)
    selected_item_ids: list[str] = field(default_factory=list)
    excluded_item_ids: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    confirmation_status: ConfirmationStatus = ConfirmationStatus.NOT_REQUESTED
    pending_action: PendingAction | None = None
    policy_evidence: list[str] = field(default_factory=list)
    facts: dict[str, ToolFact] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    consultation_only: bool = False
    last_response: str = ""

    @property
    def order_id(self) -> str | None:
        slot = self.slots.get("order_id")
        return slot.value if slot and isinstance(slot.value, str) else None

    @property
    def reason(self) -> str:
        slot = self.slots.get("reason")
        return slot.value if slot and isinstance(slot.value, str) else "no_reason_return"

    @property
    def refund_amount(self) -> float | None:
        slot = self.slots.get("refund_amount")
        if slot and isinstance(slot.value, (int, float)):
            return float(slot.value)
        return None

    def set_slot(
        self,
        name: str,
        value: Any,
        source: SlotSource,
        confidence: float,
        evidence: str,
    ) -> None:
        self.slots[name] = SlotValue(
            value=value,
            source=source,
            confidence=max(0.0, min(float(confidence), 1.0)),
            evidence=evidence,
            updated_turn=self.turn_index,
        )

    def clear_transaction_scope(self) -> None:
        self.selected_item_ids = []
        self.excluded_item_ids = []
        self.slots.pop("refund_amount", None)
        self.slots.pop("target_product_mentions", None)
        self.slots.pop("excluded_product_mentions", None)
        self.policy_evidence = []
        self.pending_action = None
        self.confirmation_status = ConfirmationStatus.NOT_REQUESTED

    def snapshot(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "turn_index": self.turn_index,
            "intent": self.intent.value,
            "stage": self.stage.value,
            "risk_level": self.risk_level.value,
            "order_id": self.order_id,
            "selected_item_ids": list(self.selected_item_ids),
            "excluded_item_ids": list(self.excluded_item_ids),
            "missing_fields": list(self.missing_fields),
            "confirmation_status": self.confirmation_status.value,
            "pending_action": asdict(self.pending_action) if self.pending_action else None,
            "consultation_only": self.consultation_only,
            "conflicts": list(self.conflicts),
            "slots": {
                name: {
                    "value": slot.value,
                    "source": slot.source.value,
                    "confidence": slot.confidence,
                    "evidence": slot.evidence,
                    "updated_turn": slot.updated_turn,
                }
                for name, slot in self.slots.items()
            },
        }

    def parser_context(self) -> dict[str, Any]:
        """Minimal state supplied to semantic parsing, without full raw history."""

        return {
            "current_intent": self.intent.value,
            "stage": self.stage.value,
            "order_id": self.order_id,
            "selected_item_ids": self.selected_item_ids,
            "excluded_item_ids": self.excluded_item_ids,
            "confirmation_pending": self.pending_action is not None,
            "missing_fields": self.missing_fields,
        }
