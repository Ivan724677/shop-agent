"""Deterministic-first router with a constrained optional model tie-breaker."""

from __future__ import annotations

from structured.models import ConfirmationStatus, Intent, SemanticFrame, StructuredTaskState

from .models import (
    ExpertName,
    ExpertResult,
    ExpertStatus,
    RouteDecision,
    RouteSource,
)
from .rules import complexity_reasons, rule_experts
from .semantic_router import SemanticRouter


class RouteProtocolError(ValueError):
    pass


class MultiAgentRouter:
    def __init__(self, semantic_router: SemanticRouter | None = None) -> None:
        self.semantic_router = semantic_router

    def decide(
        self,
        state: StructuredTaskState,
        user_message: str,
        semantic: SemanticFrame,
    ) -> RouteDecision:
        experts, reasons = rule_experts(state)
        complexity = complexity_reasons(state, user_message, semantic)
        multi_intent = (
            "MULTI_INTENT" in complexity and state.intent != Intent.HUMAN_HANDOFF
        )
        if not complexity or self.semantic_router is None:
            return RouteDecision(
                experts=[] if multi_intent else experts,
                reason_codes=list(dict.fromkeys(reasons + complexity)),
                source=RouteSource.FALLBACK if multi_intent or not experts else RouteSource.RULE,
                clarification_message=(
                    self._multi_intent_clarification()
                    if multi_intent
                    else self._clarification(state) if not experts else ""
                ),
            )

        try:
            candidate = self.semantic_router.route(
                user_message,
                self._model_state(state),
                experts,
                complexity,
            )
        except Exception as exc:
            return RouteDecision(
                experts=[] if multi_intent else experts,
                reason_codes=list(dict.fromkeys(reasons + complexity + ["MODEL_ROUTE_FAILED"])),
                source=RouteSource.FALLBACK,
                model_invoked=True,
                model_error=f"{type(exc).__name__}: {exc}",
                clarification_message=(
                    self._multi_intent_clarification()
                    if multi_intent
                    else self._clarification(state) if not experts else ""
                ),
            )

        reason_codes = list(dict.fromkeys(reasons + complexity + ["MODEL_ROUTE_REVIEWED"]))
        protected = (
            state.confirmation_status == ConfirmationStatus.CONFIRMED
            or state.pending_action is not None
            or state.intent in {Intent.REQUEST_RETURN, Intent.HUMAN_HANDOFF}
        )
        if multi_intent:
            model_handoff = (
                candidate.experts == [ExpertName.HANDOFF]
                and candidate.confidence >= 0.85
            )
            return RouteDecision(
                experts=[ExpertName.HANDOFF] if model_handoff else [],
                reason_codes=reason_codes
                + (["MODEL_SELECTED_HANDOFF"] if model_handoff else ["MULTI_INTENT_CLARIFICATION"]),
                source=RouteSource.MODEL,
                model_invoked=True,
                model_usage=candidate.usage,
                model_candidates=candidate.experts,
                model_reason=candidate.reason,
                model_confidence=candidate.confidence,
                clarification_message="" if model_handoff else self._multi_intent_clarification(),
            )
        if candidate.needs_clarification and not protected:
            return RouteDecision(
                experts=[],
                reason_codes=reason_codes + ["MODEL_REQUESTED_CLARIFICATION"],
                source=RouteSource.MODEL,
                model_invoked=True,
                model_usage=candidate.usage,
                model_candidates=candidate.experts,
                model_reason=candidate.reason,
                model_confidence=candidate.confidence,
                clarification_message=self._clarification(state),
            )

        # A model may resolve an otherwise unknown request to human support, but
        # cannot invent a transaction or remove deterministic prerequisites.
        if not experts and candidate.confidence >= 0.85 and candidate.experts == [ExpertName.HANDOFF]:
            experts = [ExpertName.HANDOFF]
            source = RouteSource.MODEL
        else:
            source = RouteSource.RULE_WITH_MODEL if experts else RouteSource.FALLBACK
        return RouteDecision(
            experts=experts,
            reason_codes=reason_codes,
            source=source,
            model_invoked=True,
            model_usage=candidate.usage,
            model_candidates=candidate.experts,
            model_reason=candidate.reason,
            model_confidence=candidate.confidence,
            clarification_message=self._clarification(state) if not experts else "",
        )

    @staticmethod
    def validate_expert_result(
        result: object,
        expected: ExpertName,
        remaining: list[ExpertName],
    ) -> ExpertResult:
        if not isinstance(result, ExpertResult):
            raise RouteProtocolError("专家返回值不是 ExpertResult。")
        if not isinstance(result.expert, ExpertName):
            raise RouteProtocolError("专家结果的 expert 字段类型无效。")
        if not isinstance(result.status, ExpertStatus):
            raise RouteProtocolError("专家结果的 status 字段类型无效。")
        if result.next_expert is not None and not isinstance(
            result.next_expert, ExpertName
        ):
            raise RouteProtocolError("专家结果的 next_expert 字段类型无效。")
        if not isinstance(result.facts, dict):
            raise RouteProtocolError("专家结果的 facts 字段类型无效。")
        if result.expert != expected:
            raise RouteProtocolError(
                f"专家身份不匹配：期望 {expected.value}，收到 {result.expert.value}。"
            )
        if result.status == ExpertStatus.CONTINUE:
            if not remaining or result.next_expert != remaining[0]:
                raise RouteProtocolError("专家请求了未授权的下一跳。")
        elif result.next_expert is not None:
            raise RouteProtocolError("终止型专家结果不能携带下一跳。")
        if result.status in {
            ExpertStatus.COMPLETE,
            ExpertStatus.CLARIFY,
            ExpertStatus.REJECTED,
        } and not result.response:
            raise RouteProtocolError("终止型专家结果缺少用户回复。")
        if result.status == ExpertStatus.ESCALATE and not result.escalation_reason:
            raise RouteProtocolError("升级结果缺少升级原因。")
        return result

    @staticmethod
    def _clarification(state: StructuredTaskState) -> str:
        if "order_id" in state.missing_fields:
            return "请提供需要处理的订单号。"
        if "target_item" in state.missing_fields:
            return "请说明需要处理的具体商品。"
        return "我可以查询订单和物流，也可以处理退货。请告诉我具体想办理什么。"

    @staticmethod
    def _multi_intent_clarification() -> str:
        return "你同时提出了多个诉求。为避免遗漏，请先确认优先处理物流查询还是退货申请。"

    @staticmethod
    def _model_state(state: StructuredTaskState) -> dict:
        """Expose only routing features; never send pending payloads or keys."""

        return {
            "intent": state.intent.value,
            "stage": state.stage.value,
            "risk_level": state.risk_level.value,
            "order_id_present": state.order_id is not None,
            "selected_item_count": len(state.selected_item_ids),
            "excluded_item_count": len(state.excluded_item_ids),
            "missing_fields": list(state.missing_fields),
            "confirmation_status": state.confirmation_status.value,
            "pending_action_present": state.pending_action is not None,
            "conflicts": list(state.conflicts),
        }
