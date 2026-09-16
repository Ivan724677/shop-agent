"""Human-handoff expert: idempotent ticket creation and reconciliation."""

from __future__ import annotations

from routing.models import ExpertName, ExpertResult, ExpertStatus
from structured.models import StructuredTaskState, TaskStage
from structured.state_machine import transition_state

from .base import BaseExpert


class HandoffExpert(BaseExpert):
    name = ExpertName.HANDOFF
    allowed_tools = frozenset({"create_ticket", "get_ticket_status"})

    def execute(self, state: StructuredTaskState, reason: str = "用户要求人工处理") -> ExpertResult:
        idempotency_key = f"ticket:{state.task_id}:{reason}"
        result = self.observe(
            state,
            "create_ticket",
            {
                "subject": "售后人工处理",
                "description": reason,
                "idempotency_key": idempotency_key,
                **({"order_id": state.order_id} if state.order_id else {}),
            },
        )
        if result.status == "UNKNOWN_COMMIT":
            result = self.observe(
                state, "get_ticket_status", {"idempotency_key": idempotency_key}
            )
        transition_state(state, TaskStage.HANDOFF)
        if result.ok and result.data.get("ticket"):
            ticket = result.data["ticket"]
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.COMPLETE,
                response=f"已创建人工工单 {ticket['ticket_id']}，请等待客服处理。",
                reason_code="HANDOFF_TICKET_CREATED",
            )
        return ExpertResult(
            expert=self.name,
            status=ExpertStatus.COMPLETE,
            response=f"暂时无法创建人工工单：{result.message}",
            reason_code="HANDOFF_TICKET_FAILED",
        )
