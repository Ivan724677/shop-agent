"""Deterministic routing rules and complexity detection."""

from __future__ import annotations

from structured.models import ConfirmationStatus, Intent, SemanticFrame, StructuredTaskState

from .models import ExpertName


def rule_experts(state: StructuredTaskState) -> tuple[list[ExpertName], list[str]]:
    if state.confirmation_status == ConfirmationStatus.REJECTED:
        return [ExpertName.TRANSACTION], ["CONFIRMATION_REJECTED"]
    if state.intent == Intent.HUMAN_HANDOFF:
        return [ExpertName.HANDOFF], ["EXPLICIT_HANDOFF"]
    if state.intent == Intent.QUERY_ORDER:
        return [ExpertName.ORDER_LOGISTICS], ["ORDER_QUERY"]
    if state.intent == Intent.QUERY_LOGISTICS:
        return [ExpertName.ORDER_LOGISTICS], ["LOGISTICS_QUERY"]
    if state.intent == Intent.REQUEST_RETURN:
        return [
            ExpertName.ORDER_LOGISTICS,
            ExpertName.POLICY,
            ExpertName.TRANSACTION,
        ], ["RETURN_WORKFLOW"]
    return [], ["AMBIGUOUS_INTENT"]


def complexity_reasons(
    state: StructuredTaskState,
    user_message: str,
    semantic: SemanticFrame,
) -> list[str]:
    reasons: list[str] = []
    if state.intent == Intent.UNKNOWN:
        reasons.append("AMBIGUOUS_INTENT")
    if state.conflicts:
        reasons.append("STATE_CONFLICT")
    if 0.0 < semantic.confidence < 0.65:
        reasons.append("LOW_SEMANTIC_CONFIDENCE")
    normalized = user_message.replace(" ", "")
    intent_groups = sum(
        (
            any(word in normalized for word in ("物流", "快递", "包裹", "到哪")),
            any(word in normalized for word in ("退货", "退款", "退掉", "能退", "申请售后")),
            any(word in normalized for word in ("人工", "客服人员", "投诉", "转接")),
        )
    )
    if intent_groups > 1:
        reasons.append("MULTI_INTENT")
    return list(dict.fromkeys(reasons))
