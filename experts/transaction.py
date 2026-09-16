"""Transaction expert: confirmation-gated return writes and reconciliation."""

from __future__ import annotations

from routing.models import ExpertName, ExpertResult, ExpertStatus
from structured.models import (
    ConfirmationStatus,
    PendingAction,
    RiskLevel,
    StructuredTaskState,
    TaskStage,
)
from structured.state_machine import transition_state

from .base import BaseExpert


class TransactionExpert(BaseExpert):
    name = ExpertName.TRANSACTION
    allowed_tools = frozenset({"create_return_request", "get_return_status"})

    def execute(self, state: StructuredTaskState) -> ExpertResult:
        if state.confirmation_status == ConfirmationStatus.REJECTED:
            state.pending_action = None
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            state.risk_level = RiskLevel.MEDIUM
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.COMPLETE,
                response="已取消刚才待确认的操作，没有创建退货申请。",
                reason_code="TRANSACTION_CANCELLED",
            )
        if not self._has_verified_scope(state):
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.ESCALATE,
                reason_code="UNVERIFIED_TRANSACTION_SCOPE",
                escalation_reason="交易前置状态不完整或政策证据缺失",
            )

        amount = float(state.refund_amount or 0)
        current_scope = (
            state.order_id,
            tuple(sorted(state.selected_item_ids)),
            round(amount, 2),
            state.reason,
        )
        if state.confirmation_status == ConfirmationStatus.CONFIRMED:
            pending = state.pending_action
            if pending is None or pending.scope_key() != current_scope:
                state.pending_action = None
                state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
                state.conflicts.append("CONFIRMATION_SCOPE_MISMATCH")
                return self._prepare_confirmation(state, amount)
            return self._execute_write(state, pending)
        return self._prepare_confirmation(state, amount)

    @staticmethod
    def _has_verified_scope(state: StructuredTaskState) -> bool:
        return bool(
            state.order_id
            and state.selected_item_ids
            and state.refund_amount is not None
            and state.refund_amount > 0
            and state.policy_evidence
            and not state.consultation_only
        )

    def _prepare_confirmation(
        self, state: StructuredTaskState, amount: float
    ) -> ExpertResult:
        item_label = "、".join(self._verified_item_names(state))
        pending = PendingAction(
            action="create_return_request",
            order_id=state.order_id or "",
            item_ids=list(state.selected_item_ids),
            amount=amount,
            reason=state.reason,
            idempotency_key=(
                f"multi-expert:{state.task_id}:{state.order_id}:"
                f"{','.join(sorted(state.selected_item_ids))}:{amount:.2f}:{state.reason}"
            ),
            requested_turn=state.turn_index,
        )
        state.pending_action = pending
        state.confirmation_status = ConfirmationStatus.PENDING
        state.risk_level = RiskLevel.HIGH
        transition_state(state, TaskStage.AWAITING_CONFIRMATION)
        excluded_text = "；已排除其他商品" if state.excluded_item_ids else ""
        return ExpertResult(
            expert=self.name,
            status=ExpertStatus.CLARIFY,
            response=(
                f"将为{item_label}申请退货，"
                f"预计退款 {amount:.2f} 元{excluded_text}。如需执行，请回复“确认退货”。"
            ),
            reason_code="CONFIRMATION_REQUIRED",
        )

    def _execute_write(
        self, state: StructuredTaskState, pending: PendingAction
    ) -> ExpertResult:
        transition_state(state, TaskStage.EXECUTING)
        state.risk_level = RiskLevel.HIGH
        creation = self.observe(
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
            reconciliation = self.observe(
                state, "get_return_status", {"idempotency_key": pending.idempotency_key}
            )
            request = reconciliation.data.get("request") if reconciliation.ok else None
            if request:
                self._complete(state)
                return ExpertResult(
                    expert=self.name,
                    status=ExpertStatus.COMPLETE,
                    response=(
                        f"写入响应丢失，但对账确认退货申请 {request['request_id']} 已创建。"
                    ),
                    reason_code="UNKNOWN_COMMIT_RECONCILED",
                )
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.ESCALATE,
                reason_code="UNKNOWN_COMMIT_UNRESOLVED",
                escalation_reason="退货操作提交状态未知",
            )
        if not creation.ok:
            return self.escalate("创建退货申请失败", creation)
        request = creation.data
        self._complete(state)
        return ExpertResult(
            expert=self.name,
            status=ExpertStatus.COMPLETE,
            response=(
                f"退货申请 {request['request_id']} 已创建，"
                f"商品为{'、'.join(self._verified_item_names(state))}，"
                f"退款金额 {request['refund_amount']:.2f} 元。"
            ),
            reason_code="RETURN_CREATED",
        )

    @staticmethod
    def _complete(state: StructuredTaskState) -> None:
        state.pending_action = None
        state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
        transition_state(state, TaskStage.COMPLETED)

    @staticmethod
    def _verified_item_names(state: StructuredTaskState) -> list[str]:
        for fact in reversed(list(state.facts.values())):
            if fact.tool != "calculate_refund" or not isinstance(fact.value, dict):
                continue
            names = [
                item.get("name")
                for item in fact.value.get("eligible_items", [])
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            ]
            if names:
                return names
        return list(state.selected_item_ids)
