import json
import unittest

from baseline.deepseek_client import ModelCompletion
from baseline.tool_catalog import SessionToolExecutor
from structured.agent import StructuredCustomerServiceAgent
from structured.models import (
    ConfirmationStatus,
    Intent,
    RiskLevel,
    SemanticFrame,
    StructuredTaskState,
    TaskStage,
)
from structured.semantic_parser import DeepSeekSemanticParser
from structured.state_machine import InvalidTransition, transition_state


class ScriptedSemanticParser:
    def __init__(self, frames: list[SemanticFrame]) -> None:
        self.frames = list(frames)
        self.contexts: list[dict] = []

    def parse(self, user_message: str, state_context: dict) -> SemanticFrame:
        self.contexts.append(state_context)
        return self.frames.pop(0) if self.frames else SemanticFrame()


class OneShotModelClient:
    def __init__(self, message: dict) -> None:
        self.message = message
        self.calls: list[dict] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelCompletion:
        self.calls.append({"messages": messages, "tools": tools})
        return ModelCompletion(message=self.message, model="fake-semantic-model")


class StructuredStateAgentTests(unittest.TestCase):
    def test_slots_and_memory_survive_multi_turn_return(self) -> None:
        agent = StructuredCustomerServiceAgent()

        first = agent.handle("我要退鞋")
        second = agent.handle("订单号是 O10086")

        self.assertIn("订单号", first)
        self.assertIn("确认退货", second)
        self.assertEqual(agent.state.order_id, "O10086")
        self.assertEqual(agent.state.selected_item_ids, ["I002"])
        self.assertEqual(agent.state.stage, TaskStage.AWAITING_CONFIRMATION)
        self.assertEqual(agent.state.risk_level, RiskLevel.HIGH)
        self.assertEqual(agent.state.missing_fields, [])
        self.assertEqual(len(agent.memory), 2)
        self.assertEqual(
            agent.state.slots["order_id"].source.value, "deterministic"
        )

    def test_confirmation_is_bound_to_pending_scope(self) -> None:
        agent = StructuredCustomerServiceAgent()

        agent.handle("我要退订单 O10086 里的鞋")
        pending_before = agent.state.pending_action
        correction = agent.handle("耳机别退")

        self.assertIsNotNone(pending_before)
        self.assertIn("确认退货", correction)
        self.assertEqual(agent.state.excluded_item_ids, ["I001"])
        self.assertEqual(agent.state.confirmation_status, ConfirmationStatus.PENDING)
        self.assertNotEqual(
            agent.state.pending_action.requested_turn, pending_before.requested_turn
        )
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)

        final = agent.handle("确认退货")
        self.assertIn("已创建", final)
        request = next(iter(agent.tool_executor.store.return_requests.values()))
        self.assertEqual(request.item_ids, ["I002"])
        self.assertEqual(agent.state.stage, TaskStage.COMPLETED)
        self.assertEqual(
            agent.state.confirmation_status, ConfirmationStatus.NOT_REQUESTED
        )

    def test_positive_target_correction_replaces_old_target(self) -> None:
        agent = StructuredCustomerServiceAgent()

        agent.handle("我要退订单 O10086 里的鞋")
        response = agent.handle("改成退耳机")

        self.assertIn("运输中", response)
        self.assertEqual(agent.state.selected_item_ids, ["I001"])
        self.assertNotIn("I002", agent.state.selected_item_ids)
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)

    def test_item_id_negative_scope_is_deterministic(self) -> None:
        agent = StructuredCustomerServiceAgent()

        response = agent.handle("订单 O10086 里 I002 退货，I001 不退")

        self.assertIn("确认退货", response)
        self.assertEqual(agent.state.selected_item_ids, ["I002"])
        self.assertEqual(agent.state.excluded_item_ids, ["I001"])

    def test_reason_change_invalidates_old_confirmation_scope(self) -> None:
        agent = StructuredCustomerServiceAgent()

        agent.handle("我要退订单 O10086 里的鞋")
        previous_pending = agent.state.pending_action
        response = agent.handle("鞋坏了，属于质量问题")

        self.assertIsNotNone(previous_pending)
        self.assertIn("确认退货", response)
        self.assertEqual(agent.state.reason, "quality_issue")
        self.assertEqual(agent.state.confirmation_status, ConfirmationStatus.PENDING)
        self.assertIsNotNone(agent.state.pending_action)
        self.assertNotEqual(
            agent.state.pending_action.idempotency_key,
            previous_pending.idempotency_key,
        )
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)

    def test_consultation_never_creates_write_or_pending_confirmation(self) -> None:
        agent = StructuredCustomerServiceAgent()

        response = agent.handle("我只是先问问，订单 O10086 的鞋能退多少钱")

        self.assertIn("只做咨询", response)
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)
        self.assertIsNone(agent.state.pending_action)
        self.assertEqual(agent.state.stage, TaskStage.COLLECTING_INFO)
        self.assertEqual(agent.state.refund_amount, 129.0)

    def test_orphan_confirmation_cannot_authorize_write(self) -> None:
        agent = StructuredCustomerServiceAgent()

        response = agent.handle("确认退货")

        self.assertIn("具体想办理什么", response)
        self.assertIn("ORPHAN_CONFIRMATION", agent.state.conflicts)
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)

    def test_deterministic_order_id_overrides_conflicting_llm_candidate(self) -> None:
        parser = ScriptedSemanticParser(
            [
                SemanticFrame(
                    intent=Intent.REQUEST_RETURN,
                    order_id="O20001",
                    product_mentions=["运动鞋"],
                    confidence=0.99,
                )
            ]
        )
        agent = StructuredCustomerServiceAgent(semantic_parser=parser)

        response = agent.handle("我要退订单 O10086 里的鞋")

        self.assertEqual(agent.state.order_id, "O10086")
        self.assertIn("ORDER_ID_LLM_DETERMINISTIC_CONFLICT", agent.state.conflicts)
        self.assertIn("确认退货", response)

    def test_llm_candidate_resolves_semantic_reference_then_policy_decides(self) -> None:
        parser = ScriptedSemanticParser(
            [
                SemanticFrame(
                    intent=Intent.REQUEST_RETURN,
                    order_id="O10086",
                    product_mentions=["运动鞋"],
                    consultation_only=True,
                    confidence=0.92,
                )
            ]
        )
        agent = StructuredCustomerServiceAgent(semantic_parser=parser)

        response = agent.handle("那个穿在脚上的商品能退吗？订单是 O10086")

        self.assertIn("只做咨询", response)
        self.assertEqual(agent.state.selected_item_ids, ["I002"])
        self.assertEqual(agent.state.order_id, "O10086")
        self.assertEqual(len(agent.tool_executor.store.return_requests), 0)

    def test_relative_order_candidate_is_not_accepted_without_grounding(self) -> None:
        parser = ScriptedSemanticParser(
            [
                SemanticFrame(
                    intent=Intent.REQUEST_RETURN,
                    order_id="O10086",
                    product_mentions=["运动鞋"],
                    confidence=0.95,
                )
            ]
        )
        agent = StructuredCustomerServiceAgent(semantic_parser=parser)

        response = agent.handle("上一单的鞋帮我退一下")

        self.assertIn("订单号", response)
        self.assertIsNone(agent.state.order_id)
        self.assertIn(
            "RELATIVE_ORDER_REQUIRES_GROUNDED_RESOLUTION", agent.state.conflicts
        )

    def test_tool_timeout_after_commit_is_reconciled(self) -> None:
        executor = SessionToolExecutor(user_id="U001")
        executor.gateway.inject_failure_once(
            "create_return_request", "TIMEOUT_AFTER_COMMIT"
        )
        agent = StructuredCustomerServiceAgent(tool_executor=executor)

        agent.handle("我要退订单 O10086 里的鞋")
        response = agent.handle("确认退货")

        self.assertIn("对账确认", response)
        self.assertEqual(len(executor.store.return_requests), 1)
        tool_names = [
            event.detail["tool"]
            for event in agent.trace
            if event.event_type == "tool"
        ]
        self.assertIn("get_return_status", tool_names)

    def test_memory_is_bounded_but_working_state_remains(self) -> None:
        agent = StructuredCustomerServiceAgent(max_memory_turns=2)

        agent.handle("查订单")
        agent.handle("O10086")
        agent.handle("再查一下 O10086")

        self.assertEqual(len(agent.memory), 2)
        self.assertEqual(agent.state.order_id, "O10086")
        self.assertGreaterEqual(agent.state.turn_index, 3)

    def test_state_machine_rejects_illegal_jump(self) -> None:
        state = StructuredTaskState(user_id="U001")
        with self.assertRaises(InvalidTransition):
            transition_state(state, TaskStage.COMPLETED)

    def test_session_identity_applies_to_policy_and_refund_tools(self) -> None:
        executor = SessionToolExecutor(user_id="U001")

        policy = executor.execute(
            "search_policy",
            {"order_id": "O20001", "item_id": "I004", "reason": "no_reason_return"},
        )

        self.assertFalse(policy.ok)
        self.assertEqual(policy.error_code, "FORBIDDEN")


class DeepSeekSemanticParserTests(unittest.TestCase):
    def test_parser_reads_function_arguments_into_semantic_frame(self) -> None:
        model = OneShotModelClient(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "parse_1",
                        "type": "function",
                        "function": {
                            "name": "emit_semantic_frame",
                            "arguments": json.dumps(
                                {
                                    "intent": "request_return",
                                    "order_id": "O10086",
                                    "item_ids": [],
                                    "product_mentions": ["运动鞋"],
                                    "excluded_item_ids": [],
                                    "excluded_product_mentions": ["蓝牙耳机"],
                                    "reason": "no_reason_return",
                                    "consultation_only": False,
                                    "confirmation": "none",
                                    "wants_handoff": False,
                                    "confidence": 0.93,
                                }
                            ),
                        },
                    }
                ],
            }
        )
        parser = DeepSeekSemanticParser(model)

        frame = parser.parse("退鞋，耳机不退", {"current_intent": "unknown"})

        self.assertEqual(frame.intent, Intent.REQUEST_RETURN)
        self.assertEqual(frame.product_mentions, ["运动鞋"])
        self.assertEqual(frame.excluded_product_mentions, ["蓝牙耳机"])
        self.assertEqual(model.calls[0]["tools"][0]["function"]["name"], "emit_semantic_frame")


if __name__ == "__main__":
    unittest.main(verbosity=2)
