"""Policy expert: evidence retrieval and deterministic eligibility calculation."""

from __future__ import annotations

from routing.models import ExpertName, ExpertResult, ExpertStatus
from rag.agent import AgenticRAG
from structured.models import ConfirmationStatus, RiskLevel, SlotSource, StructuredTaskState, TaskStage, ToolFact
from structured.state_machine import transition_state

from .base import BaseExpert


class PolicyExpert(BaseExpert):
    name = ExpertName.POLICY
    allowed_tools = frozenset({"search_policy", "calculate_refund"})

    def __init__(self, tool_caller, rag_agent: AgenticRAG | None = None) -> None:
        super().__init__(tool_caller)
        self.rag_agent = rag_agent

    def execute(self, state: StructuredTaskState) -> ExpertResult:
        if not state.order_id or not state.selected_item_ids:
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.ESCALATE,
                reason_code="UNGROUNDED_POLICY_SCOPE",
                escalation_reason="政策专家收到未完成实体校验的任务",
            )
        transition_state(state, TaskStage.CHECKING_POLICY)
        if self.rag_agent is not None:
            rag_check = self._agentic_policy_check(state)
            if rag_check is not None:
                return rag_check
        evidence: list[str] = []
        for item_id in state.selected_item_ids:
            result = self.observe(
                state,
                "search_policy",
                {"order_id": state.order_id, "item_id": item_id, "reason": state.reason},
            )
            if not result.ok:
                return self.escalate("政策检索失败", result)
            evidence.extend(doc["id"] for doc in result.data.get("evidence", []))
        state.policy_evidence = sorted(set(evidence))
        if not state.policy_evidence:
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.ESCALATE,
                reason_code="POLICY_EVIDENCE_MISSING",
                escalation_reason="政策检索没有返回可追溯证据",
            )

        calculation = self.observe(
            state,
            "calculate_refund",
            {
                "order_id": state.order_id,
                "item_ids": state.selected_item_ids,
                "reason": state.reason,
            },
        )
        if not calculation.ok:
            return self.escalate("退款资格计算失败", calculation)
        ineligible = calculation.data.get("ineligible_items", [])
        if ineligible:
            state.pending_action = None
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            transition_state(state, TaskStage.REJECTED)
            reasons = "；".join(item["reason"] for item in ineligible)
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.REJECTED,
                response=f"当前不能自动退货：{reasons}",
                reason_code="POLICY_REJECTED",
            )

        amount = float(calculation.data["refund_amount"])
        state.set_slot("refund_amount", amount, SlotSource.TOOL, 1.0, "calculate_refund")
        state.missing_fields = []
        item_names = [item["name"] for item in calculation.data.get("eligible_items", [])]
        if state.consultation_only:
            state.pending_action = None
            state.confirmation_status = ConfirmationStatus.NOT_REQUESTED
            state.risk_level = RiskLevel.MEDIUM
            transition_state(state, TaskStage.COLLECTING_INFO)
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.COMPLETE,
                response=(
                    f"经核验，{'、'.join(item_names)}预计可退 {amount:.2f} 元。"
                    "这次只做咨询，没有创建退货申请。"
                ),
                reason_code="CONSULTATION_COMPLETED",
                facts={"item_names": item_names, "refund_amount": amount},
            )
        return ExpertResult(
            expert=self.name,
            status=ExpertStatus.CONTINUE,
            next_expert=ExpertName.TRANSACTION,
            reason_code="POLICY_ELIGIBLE",
            facts={"item_names": item_names, "refund_amount": amount},
        )

    def _agentic_policy_check(self, state: StructuredTaskState) -> ExpertResult | None:
        product_type = "standard"
        item_name = "商品"
        for fact in reversed(list(state.facts.values())):
            if fact.tool != "get_order" or not isinstance(fact.value, dict):
                continue
            order = fact.value.get("order", {})
            for item in order.get("items", []):
                if item.get("item_id") in state.selected_item_ids:
                    product_type = "custom" if item.get("is_custom") else "standard"
                    item_name = str(item.get("name", item_name))
                    break
            break
        result = self.rag_agent.run(
            f"{product_type} 商品 {item_name} {state.reason} 退货政策资格和期限",
            metadata_filter={"product_type": product_type, "reason": state.reason},
            required_aspects=[product_type, "return"],
        )
        state.facts[f"agentic_rag:{len(state.facts) + 1}"] = ToolFact(
            key="agentic_rag",
            value={
                "status": result.status,
                "evidence_ids": result.evidence_ids,
                "validation": result.validation.as_dict(),
                "retrieval_count": result.retrieval_count,
                "grader_calls": result.grader_calls,
                "rewrite_count": result.rewrite_count,
            },
            tool="agentic_rag",
            observed_turn=state.turn_index,
        )
        if result.status == "UNCERTAIN":
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.ESCALATE,
                reason_code="AGENTIC_RAG_UNCERTAIN",
                escalation_reason="Agentic RAG 无法确认政策证据：" + ",".join(result.validation.reason_codes),
            )
        if result.status not in {"GROUNDED", "NO_RETRIEVAL"}:
            return ExpertResult(
                expert=self.name,
                status=ExpertStatus.ESCALATE,
                reason_code="AGENTIC_RAG_NOT_GROUNDED",
                escalation_reason="政策回答没有通过 Agentic RAG 证据校验",
            )
        return None
