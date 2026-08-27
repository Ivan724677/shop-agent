"""状态归并器（阶段三决策层第一环，负责"状态决策"）。

把确定性提取器与 LLM 语义解析器的候选合并进 StructuredTaskState，
并强制执行安全规则：
- 权威性：确定性 > LLM，冲突记录不静默覆盖
- 相对指代（"上一单"）不落地，必须用户提供可验证订单号
- 新任务判定：意图变更或终态重启 → 换 task_id 并清空交易范围
- 范围变更（订单/商品/原因）→ 作废 pending_action 与旧确认
- 确认判定：reject / 孤儿确认 / 范围变更确认都不作数，仅合法确认置 CONFIRMED
- 派生字段：每轮重算 missing_fields 与 risk_level

是"LLM 做候选、代码定权威"这一设计主张的核心实现。
"""

from __future__ import annotations

import uuid

from .models import (
    ConfirmationStatus,
    DeterministicSignals,
    Intent,
    RiskLevel,
    SemanticFrame,
    SlotSource,
    StructuredTaskState,
    TaskStage,
)
from .state_machine import transition_state


class StateReducer:
    def reduce(
        self,
        state: StructuredTaskState,
        user_message: str,
        deterministic: DeterministicSignals,
        semantic: SemanticFrame,
    ) -> None:
        state.turn_index += 1
        state.conflicts = []
        effective_intent = self._effective_intent(deterministic, semantic)
        self._maybe_start_new_task(state, effective_intent)

        if effective_intent not in {Intent.UNKNOWN, Intent.KEEP_CURRENT}:
            state.intent = effective_intent

        scope_changed = self._merge_order(state, user_message, deterministic, semantic)
        scope_changed |= self._merge_targets(state, user_message, deterministic, semantic)
        scope_changed |= self._merge_reason(
            state, user_message, deterministic, semantic
        )

        state.consultation_only = (
            deterministic.consultation_only or semantic.consultation_only
        )
        if deterministic.relative_order_reference and state.order_id is None:
            state.conflicts.append("AMBIGUOUS_RELATIVE_ORDER_REFERENCE")

        confirmation = (
            deterministic.confirmation
            if deterministic.confirmation != "none"
            else semantic.confirmation
        )
        self._merge_confirmation(state, confirmation, scope_changed)

        if semantic.wants_handoff or state.intent == Intent.HUMAN_HANDOFF:
            state.intent = Intent.HUMAN_HANDOFF
            state.risk_level = RiskLevel.MEDIUM
            state.missing_fields = []
            transition_state(state, TaskStage.HANDOFF)
            return

        self.recompute_derived(state)

    @staticmethod
    def _effective_intent(
        deterministic: DeterministicSignals, semantic: SemanticFrame
    ) -> Intent:
        if deterministic.explicit_intent != Intent.UNKNOWN:
            return deterministic.explicit_intent
        return semantic.intent

    @staticmethod
    def _maybe_start_new_task(state: StructuredTaskState, intent: Intent) -> None:
        if intent in {Intent.UNKNOWN, Intent.KEEP_CURRENT}:
            return
        intent_changed = state.intent not in {Intent.UNKNOWN, intent}
        terminal_restart = state.stage in {
            TaskStage.COMPLETED,
            TaskStage.REJECTED,
            TaskStage.HANDOFF,
        }
        if intent_changed or terminal_restart:
            preserved_order = state.slots.get("order_id")
            state.task_id = str(uuid.uuid4())
            state.clear_transaction_scope()
            #保留order_id槽位（大概率还用一个订单）
            state.slots = {"order_id": preserved_order} if preserved_order else {}
            state.facts = {}
            state.missing_fields = []
            state.consultation_only = False
            transition_state(state, TaskStage.COLLECTING_INFO)

    @staticmethod
    def _merge_order(
        state: StructuredTaskState,
        user_message: str,
        deterministic: DeterministicSignals,
        semantic: SemanticFrame,
    ) -> bool:
        if (
            deterministic.relative_order_reference
            and deterministic.order_id is None
            and semantic.order_id is not None
        ):
            state.conflicts.append("RELATIVE_ORDER_REQUIRES_GROUNDED_RESOLUTION")
            return False
        chosen = deterministic.order_id or semantic.order_id
        if not chosen:
            return False
        chosen = chosen.upper()
        if deterministic.order_id and semantic.order_id and deterministic.order_id != semantic.order_id.upper():
            state.conflicts.append("ORDER_ID_LLM_DETERMINISTIC_CONFLICT")
        changed = state.order_id is not None and state.order_id != chosen
        if changed:
            state.clear_transaction_scope()
            state.facts = {}
        source = SlotSource.DETERMINISTIC if deterministic.order_id else SlotSource.LLM
        confidence = 1.0 if deterministic.order_id else max(semantic.confidence, 0.5)
        state.set_slot("order_id", chosen, source, confidence, user_message)
        return changed

    @staticmethod
    def _merge_targets(
        state: StructuredTaskState,
        user_message: str,
        deterministic: DeterministicSignals,
        semantic: SemanticFrame,
    ) -> bool:
        item_ids = deterministic.item_ids or semantic.item_ids
        excluded_item_ids = deterministic.excluded_item_ids or semantic.excluded_item_ids
        mentions = deterministic.product_mentions or semantic.product_mentions
        excluded_mentions = (
            deterministic.excluded_product_mentions or semantic.excluded_product_mentions
        )
        has_scope_signal = bool(item_ids or excluded_item_ids or mentions or excluded_mentions)
        if not has_scope_signal:
            return False

        before = (
            tuple(state.selected_item_ids),
            tuple(state.excluded_item_ids),
            tuple(state.slots.get("target_product_mentions", SlotProxy([])).value),
            tuple(state.slots.get("excluded_product_mentions", SlotProxy([])).value),
        )
        source = (
            SlotSource.DETERMINISTIC
            if deterministic.item_ids
            or deterministic.product_mentions
            or deterministic.excluded_product_mentions
            else SlotSource.LLM
        )
        confidence = 1.0 if source == SlotSource.DETERMINISTIC else max(semantic.confidence, 0.5)
        if item_ids:
            state.selected_item_ids = sorted(set(item_ids) - set(excluded_item_ids))
        if excluded_item_ids:
            state.excluded_item_ids = sorted(
                set(state.excluded_item_ids) | set(excluded_item_ids)
            )
            state.selected_item_ids = sorted(
                set(state.selected_item_ids) - set(state.excluded_item_ids)
            )
        if mentions:
            # A new positive product reference replaces the previously resolved
            # target scope. Negative-only corrections are handled separately and
            # intentionally retain the remaining selected items.
            if not item_ids:
                state.selected_item_ids = []
            state.set_slot(
                "target_product_mentions",
                sorted(set(mentions) - set(excluded_mentions)),
                source,
                confidence,
                user_message,
            )
        if excluded_mentions:
            existing = state.slots.get("excluded_product_mentions")
            combined = set(existing.value if existing else []) | set(excluded_mentions)
            state.set_slot(
                "excluded_product_mentions",
                sorted(combined),
                source,
                confidence,
                user_message,
            )
            target_slot = state.slots.get("target_product_mentions")
            if target_slot:
                target_slot.value = sorted(set(target_slot.value) - combined)
        after = (
            tuple(state.selected_item_ids),
            tuple(state.excluded_item_ids),
            tuple(state.slots.get("target_product_mentions", SlotProxy([])).value),
            tuple(state.slots.get("excluded_product_mentions", SlotProxy([])).value),
        )
        changed = before != after
        if changed and state.pending_action:
            state.pending_action = None
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            state.slots.pop("refund_amount", None)
            state.policy_evidence = []
        return changed

    @staticmethod
    def _merge_reason(
        state: StructuredTaskState,
        user_message: str,
        deterministic: DeterministicSignals,
        semantic: SemanticFrame,
    ) -> bool:
        reason = deterministic.reason or semantic.reason
        if not reason:
            if "reason" not in state.slots and state.intent == Intent.REQUEST_RETURN:
                state.set_slot(
                    "reason",
                    "no_reason_return",
                    SlotSource.DERIVED,
                    0.8,
                    "default_return_reason",
                )
            return False
        previous = state.reason if "reason" in state.slots else None
        source = SlotSource.DETERMINISTIC if deterministic.reason else SlotSource.LLM
        state.set_slot(
            "reason",
            reason,
            source,
            1.0 if deterministic.reason else max(semantic.confidence, 0.5),
            user_message,
        )
        changed = previous is not None and previous != reason
        if changed and state.pending_action:
            state.pending_action = None
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            state.slots.pop("refund_amount", None)
            state.policy_evidence = []
        return changed

    @staticmethod
    def _merge_confirmation(
        state: StructuredTaskState, confirmation: str, scope_changed: bool
    ) -> None:
        if confirmation == "reject":
            state.pending_action = None
            state.confirmation_status = ConfirmationStatus.REJECTED
            if state.stage == TaskStage.AWAITING_CONFIRMATION:
                transition_state(state, TaskStage.COLLECTING_INFO)
            return
        if confirmation != "confirm":
            return
        if scope_changed:
            state.conflicts.append("SCOPE_CHANGED_REQUIRES_RECONFIRMATION")
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            return
        if state.pending_action is None:
            state.conflicts.append("ORPHAN_CONFIRMATION")
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            return
        state.confirmation_status = ConfirmationStatus.CONFIRMED

    @staticmethod
    def recompute_derived(state: StructuredTaskState) -> None:
        missing: list[str] = []
        if state.intent in {Intent.QUERY_ORDER, Intent.QUERY_LOGISTICS, Intent.REQUEST_RETURN}:
            if not state.order_id:
                missing.append("order_id")
        if state.intent == Intent.REQUEST_RETURN:
            target_mentions = state.slots.get("target_product_mentions")
            if not state.selected_item_ids and not (target_mentions and target_mentions.value):
                missing.append("target_item")
            state.risk_level = (
                RiskLevel.HIGH
                if state.pending_action or state.confirmation_status == ConfirmationStatus.CONFIRMED
                else RiskLevel.MEDIUM
            )
        elif state.intent in {Intent.QUERY_ORDER, Intent.QUERY_LOGISTICS}:
            state.risk_level = RiskLevel.LOW
        else:
            state.risk_level = RiskLevel.LOW
        state.missing_fields = missing
        if state.stage == TaskStage.IDLE:
            transition_state(state, TaskStage.COLLECTING_INFO)


class SlotProxy:
    """Internal default object used only to simplify tuple comparisons."""

    def __init__(self, value: object) -> None:
        self.value = value
