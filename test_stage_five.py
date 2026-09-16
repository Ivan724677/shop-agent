import json
import unittest
from pathlib import Path

from baseline.deepseek_client import ModelCompletion
from baseline.tool_catalog import SessionToolExecutor
from experts.order_logistics import OrderLogisticsExpert
from evaluation.comparison import compare_variants
from multi_agent.agent import MultiExpertCustomerServiceAgent
from routing.models import ExpertName, ExpertResult, ExpertStatus
from routing.router import MultiAgentRouter
from routing.semantic_router import DeepSeekSemanticRouter
from scenarios.loader import load_file
from structured.models import TaskStage


class ScriptedRouteModel:
    def __init__(
        self,
        experts: list[str],
        *,
        confidence: float = 0.9,
        needs_clarification: bool = False,
    ) -> None:
        self.experts = experts
        self.confidence = confidence
        self.needs_clarification = needs_clarification
        self.calls: list[dict] = []

    def complete(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        return ModelCompletion(
            message={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "route_1",
                        "type": "function",
                        "function": {
                            "name": "emit_route_decision",
                            "arguments": json.dumps(
                                {
                                    "experts": self.experts,
                                    "needs_clarification": self.needs_clarification,
                                    "reason": "scripted complex routing",
                                    "confidence": self.confidence,
                                }
                            ),
                        },
                    }
                ],
            },
            usage={"prompt_tokens": 20, "completion_tokens": 5},
            model="fake-router",
        )


class FailingSemanticRouter:
    def route(self, *args, **kwargs):
        raise RuntimeError("router unavailable")


class InvalidPolicyExpert:
    name = ExpertName.POLICY

    def execute(self, state):
        return {"status": "complete", "response": "未经验证的成功"}


class InvalidNextHopPolicyExpert:
    name = ExpertName.POLICY

    def execute(self, state):
        return ExpertResult(
            expert=ExpertName.POLICY,
            status=ExpertStatus.CONTINUE,
            next_expert=ExpertName.HANDOFF,
        )


class InvalidStatusPolicyExpert:
    name = ExpertName.POLICY

    def execute(self, state):
        return ExpertResult(
            expert=ExpertName.POLICY,
            status="complete",
            response="伪造成功",
        )


class StageFiveRoutingTests(unittest.TestCase):
    def test_order_query_routes_only_to_order_expert(self) -> None:
        agent = MultiExpertCustomerServiceAgent()

        response = agent.handle("查询订单 O10086")

        self.assertIn("partially_delivered", response)
        self.assertEqual(agent.route_trace[0]["experts"], ["order_logistics"])
        self.assertEqual(agent.route_trace[0]["final_expert"], "order_logistics")
        self.assertEqual(self._tools(agent), ["get_order"])

    def test_logistics_query_routes_to_order_expert(self) -> None:
        agent = MultiExpertCustomerServiceAgent()

        response = agent.handle("订单 O10086 的物流到哪了")

        self.assertIn("SF10086001", response)
        self.assertEqual(self._experts(agent), ["order_logistics"])
        self.assertEqual(self._tools(agent), ["get_order", "list_shipments"])

    def test_return_uses_order_policy_transaction_chain(self) -> None:
        agent = MultiExpertCustomerServiceAgent()

        response = agent.handle("我要退订单 O10086 里的鞋")

        self.assertIn("确认退货", response)
        self.assertEqual(
            self._experts(agent), ["order_logistics", "policy", "transaction"]
        )
        self.assertEqual(
            self._tools(agent), ["get_order", "search_policy", "calculate_refund"]
        )
        self.assertTrue(agent.state.policy_evidence)
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)

    def test_transaction_writes_only_after_scope_bound_confirmation(self) -> None:
        agent = MultiExpertCustomerServiceAgent()

        agent.handle("订单 O10086 的 I002 退货")
        first_pending = agent.state.pending_action
        self.assertIsNotNone(first_pending)
        self.assertNotIn("create_return_request", self._tools(agent))

        response = agent.handle("确认退货")

        self.assertIn("已创建", response)
        self.assertEqual(len(agent.tool_executor.store.return_requests), 1)
        request = next(iter(agent.tool_executor.store.return_requests.values()))
        self.assertEqual(request.item_ids, ["I002"])
        self.assertEqual(agent.state.stage, TaskStage.COMPLETED)

    def test_consultation_stops_at_policy_expert(self) -> None:
        agent = MultiExpertCustomerServiceAgent()

        response = agent.handle("我只是问问订单 O10086 的鞋能退多少钱")

        self.assertIn("只做咨询", response)
        self.assertEqual(agent.route_trace[0]["final_expert"], "policy")
        self.assertNotIn("transaction", self._experts(agent))
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)

    def test_explicit_handoff_routes_directly_to_handoff_expert(self) -> None:
        agent = MultiExpertCustomerServiceAgent()

        response = agent.handle("订单 O10086 我要转人工处理")

        self.assertIn("人工工单", response)
        self.assertEqual(self._experts(agent), ["handoff"])
        self.assertEqual(self._tools(agent), ["create_ticket"])

    def test_multi_intent_invokes_model_and_clarifies_without_write(self) -> None:
        model = ScriptedRouteModel(["order_logistics", "transaction"])
        router = MultiAgentRouter(DeepSeekSemanticRouter(model))
        agent = MultiExpertCustomerServiceAgent(router=router)

        response = agent.handle("查一下订单 O10086 的物流，同时把鞋退掉")

        self.assertIn("多个诉求", response)
        self.assertTrue(agent.route_trace[0]["model_invoked"])
        self.assertEqual(
            agent.route_trace[0]["model_candidates"],
            ["order_logistics", "transaction"],
        )
        self.assertEqual(self._tools(agent), [])
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)

    def test_model_cannot_route_unknown_request_to_transaction(self) -> None:
        model = ScriptedRouteModel(["transaction"], confidence=0.99)
        agent = MultiExpertCustomerServiceAgent(
            router=MultiAgentRouter(DeepSeekSemanticRouter(model))
        )

        response = agent.handle("帮我处理一下这个事情")

        self.assertIn("具体想办理什么", response)
        self.assertEqual(agent.route_trace[0]["experts"], [])
        self.assertEqual(self._tools(agent), [])

    def test_high_confidence_model_can_route_unknown_request_to_handoff(self) -> None:
        model = ScriptedRouteModel(["handoff"], confidence=0.95)
        agent = MultiExpertCustomerServiceAgent(
            router=MultiAgentRouter(DeepSeekSemanticRouter(model))
        )

        response = agent.handle("这个问题需要特殊处理")

        self.assertIn("人工工单", response)
        self.assertEqual(agent.route_trace[0]["source"], "model")
        self.assertEqual(self._experts(agent), ["handoff"])

    def test_model_failure_has_deterministic_fallback(self) -> None:
        agent = MultiExpertCustomerServiceAgent(
            router=MultiAgentRouter(FailingSemanticRouter())
        )

        response = agent.handle("暂时说不清楚")

        self.assertIn("具体想办理什么", response)
        self.assertEqual(agent.route_trace[0]["source"], "fallback")
        self.assertIn("MODEL_ROUTE_FAILED", agent.route_trace[0]["reason_codes"])
        self.assertIn("router unavailable", agent.route_trace[0]["model_error"])

    def test_unresolved_unknown_commit_escalates_to_handoff(self) -> None:
        executor = SessionToolExecutor(user_id="U001")
        executor.gateway.inject_failure_once(
            "create_return_request", "TIMEOUT_AFTER_COMMIT"
        )
        executor.gateway.inject_failures("get_return_status", "TIMEOUT", "TIMEOUT")
        agent = MultiExpertCustomerServiceAgent(tool_executor=executor)

        agent.handle("订单 O10086 的鞋退货")
        response = agent.handle("确认退货")

        self.assertIn("人工工单", response)
        self.assertEqual(len(executor.store.return_requests), 1)
        self.assertEqual(len(executor.store.tickets), 1)
        self.assertEqual(agent.route_trace[-1]["final_expert"], "handoff")
        self.assertIn("提交状态未知", agent.route_trace[-1]["fallback_reason"])

    def test_invalid_expert_schema_is_rejected_and_escalated(self) -> None:
        agent = MultiExpertCustomerServiceAgent()
        agent.experts[ExpertName.POLICY] = InvalidPolicyExpert()

        response = agent.handle("订单 O10086 的鞋退货")

        self.assertIn("人工工单", response)
        self.assertIn("EXPERT_PROTOCOL_ERROR", agent.route_trace[0]["fallback_reason"])
        self.assertNotIn("create_return_request", self._tools(agent))

    def test_invalid_expert_next_hop_is_rejected(self) -> None:
        agent = MultiExpertCustomerServiceAgent()
        agent.experts[ExpertName.POLICY] = InvalidNextHopPolicyExpert()

        response = agent.handle("订单 O10086 的鞋退货")

        self.assertIn("人工工单", response)
        self.assertIn("未授权的下一跳", agent.route_trace[0]["fallback_reason"])

    def test_dataclass_with_invalid_runtime_field_type_is_rejected(self) -> None:
        agent = MultiExpertCustomerServiceAgent()
        agent.experts[ExpertName.POLICY] = InvalidStatusPolicyExpert()

        response = agent.handle("订单 O10086 的鞋退货")

        self.assertIn("人工工单", response)
        self.assertIn("status 字段类型无效", agent.route_trace[0]["fallback_reason"])

    def test_model_receives_minimal_state_without_pending_payload(self) -> None:
        model = ScriptedRouteModel(["order_logistics", "transaction"])
        agent = MultiExpertCustomerServiceAgent(
            router=MultiAgentRouter(DeepSeekSemanticRouter(model))
        )
        agent.handle("订单 O10086 的鞋退货")

        response = agent.handle("再查物流并退货")

        self.assertIn("多个诉求", response)
        model_payload = json.loads(model.calls[0]["messages"][1]["content"])
        routed_state = model_payload["structured_state"]
        self.assertTrue(routed_state["pending_action_present"])
        self.assertNotIn("pending_action", routed_state)
        self.assertNotIn("idempotency_key", json.dumps(routed_state))

    def test_experts_share_mcp_audit_boundary_and_route_trace(self) -> None:
        agent = MultiExpertCustomerServiceAgent()

        agent.handle("订单 O10086 的鞋退货")

        audit_tools = [event.tool for event in agent.tool_executor.audit_log]
        self.assertEqual(audit_tools, ["get_order", "search_policy", "calculate_refund"])
        self.assertTrue(all(event.actor_id == "U001" for event in agent.tool_executor.audit_log))
        tool_events = [event for event in agent.trace if event.event_type == "tool"]
        self.assertTrue(all(event.detail["audit_id"] for event in tool_events))
        self.assertEqual(agent.route_trace[0]["reason_codes"], ["RETURN_WORKFLOW"])

    def test_expert_tool_allowlist_blocks_cross_domain_call(self) -> None:
        executor = SessionToolExecutor(user_id="U001")
        expert = OrderLogisticsExpert(executor.execute)
        agent = MultiExpertCustomerServiceAgent(tool_executor=executor)

        result = expert.observe(
            agent.state,
            "create_return_request",
            {
                "order_id": "O10086",
                "item_ids": ["I002"],
                "refund_amount": 129.0,
                "reason": "no_reason_return",
                "idempotency_key": "forbidden-expert-call",
                "user_confirmation": True,
            },
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "EXPERT_TOOL_FORBIDDEN")
        self.assertEqual(executor.audit_log, [])
        self.assertEqual(len(executor.store.return_requests), 0)

    def test_multi_expert_variant_runs_in_existing_comparison_harness(self) -> None:
        scenario = load_file(Path("scenarios/golden/confirmation_required.json"))

        def factory(user_id: str):
            executor = SessionToolExecutor(user_id=user_id)
            return MultiExpertCustomerServiceAgent(
                tool_executor=executor, user_id=user_id
            ), executor

        results, summaries = compare_variants([scenario], {"multi_expert": factory})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].unsafe_writes, [])
        self.assertEqual(summaries[0].variant, "multi_expert")

    @staticmethod
    def _tools(agent: MultiExpertCustomerServiceAgent) -> list[str]:
        return [
            event.detail["tool"]
            for event in agent.trace
            if event.event_type == "tool"
        ]

    @staticmethod
    def _experts(agent: MultiExpertCustomerServiceAgent) -> list[str]:
        return [
            event.detail["expert"]
            for event in agent.trace
            if event.event_type == "expert"
        ]


if __name__ == "__main__":
    unittest.main(verbosity=2)
