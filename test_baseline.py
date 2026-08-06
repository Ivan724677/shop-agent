import json
import os
import unittest
from unittest.mock import patch

from baseline.deepseek_client import (
    DeepSeekClient,
    DeepSeekConfigurationError,
    ModelCompletion,
)
from baseline.react_agent import ReActBaselineAgent
from baseline.tool_catalog import BASELINE_TOOLS, SessionToolExecutor


def tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class ScriptedModelClient:
    def __init__(self, messages: list[dict]) -> None:
        self.responses = [
            ModelCompletion(message=message, finish_reason="tool_calls", model="fake-model")
            for message in messages
        ]
        self.calls: list[dict] = []

    def complete(self, messages: list[dict], tools: list[dict]) -> ModelCompletion:
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            raise AssertionError("fake model response exhausted")
        response = self.responses.pop(0)
        if not response.message.get("tool_calls"):
            response.finish_reason = "stop"
        return response


class ReActBaselineTests(unittest.TestCase):
    def test_model_selects_tool_observes_result_then_answers(self) -> None:
        model = ScriptedModelClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_1", "get_order", '{"order_id":"O10086"}')],
                },
                {
                    "role": "assistant",
                    "content": "订单 O10086 包含蓝牙耳机和运动鞋。",
                },
            ]
        )
        agent = ReActBaselineAgent(model)

        answer = agent.handle("订单 O10086 有哪些商品？")

        self.assertIn("运动鞋", answer)
        self.assertEqual(
            [message["role"] for message in agent.messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        observation = json.loads(agent.messages[3]["content"])
        self.assertTrue(observation["ok"])
        self.assertEqual(observation["data"]["order"]["order_id"], "O10086")
        self.assertEqual(agent.trace[1].detail["tool"], "get_order")

    def test_multi_step_tool_chain_is_supported(self) -> None:
        model = ScriptedModelClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_1", "list_orders", "{}")],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call("call_2", "list_shipments", '{"order_id":"O10086"}')
                    ],
                },
                {
                    "role": "assistant",
                    "content": "订单 O10086 分为两个包裹，其中一个运输中，一个已签收。",
                },
            ]
        )
        agent = ReActBaselineAgent(model)

        answer = agent.handle("我最近那笔订单物流怎么样？")

        self.assertIn("两个包裹", answer)
        self.assertEqual([event.event_type for event in agent.trace], ["model", "tool", "model", "tool", "model"])

    def test_invalid_arguments_become_observation_instead_of_crash(self) -> None:
        model = ScriptedModelClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("bad_call", "get_order", "{not-json")],
                },
                {
                    "role": "assistant",
                    "content": "订单参数解析失败，请重新提供订单号。",
                },
            ]
        )
        agent = ReActBaselineAgent(model)

        answer = agent.handle("查订单")

        observation = json.loads(agent.messages[3]["content"])
        self.assertEqual(observation["error"]["code"], "INVALID_ARGUMENTS_JSON")
        self.assertIn("重新提供", answer)

    def test_model_cannot_override_authenticated_user(self) -> None:
        model = ScriptedModelClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "call_1",
                            "get_order",
                            '{"order_id":"O20001","user_id":"U002"}',
                        )
                    ],
                },
                {"role": "assistant", "content": "我无法访问该订单。"},
            ]
        )
        agent = ReActBaselineAgent(
            model, tool_executor=SessionToolExecutor(user_id="U001")
        )

        agent.handle("查询订单 O20001")

        observation = json.loads(agent.messages[3]["content"])
        self.assertFalse(observation["ok"])
        self.assertEqual(observation["error"]["code"], "IDENTITY_OVERRIDE_ATTEMPT")

    def test_repository_still_blocks_unconfirmed_high_risk_write(self) -> None:
        model = ScriptedModelClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "call_1",
                            "create_return_request",
                            json.dumps(
                                {
                                    "order_id": "O10086",
                                    "item_ids": ["I002"],
                                    "refund_amount": 129.0,
                                    "reason": "no_reason_return",
                                    "idempotency_key": "baseline-test",
                                    "user_confirmation": False,
                                }
                            ),
                        )
                    ],
                },
                {"role": "assistant", "content": "需要你明确确认后才能创建退货申请。"},
            ]
        )
        executor = SessionToolExecutor(user_id="U001")
        agent = ReActBaselineAgent(model, tool_executor=executor)

        answer = agent.handle("我考虑退鞋")

        observation = json.loads(agent.messages[3]["content"])
        self.assertEqual(observation["error"]["code"], "CONFIRMATION_REQUIRED")
        self.assertEqual(len(executor.store.return_requests), 0)
        self.assertIn("明确确认", answer)

    def test_unexpected_tool_argument_becomes_validation_observation(self) -> None:
        model = ScriptedModelClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        tool_call(
                            "call_1",
                            "get_order",
                            '{"order_id":"O10086","unexpected":"value"}',
                        )
                    ],
                },
                {"role": "assistant", "content": "工具参数有误，我需要重新查询。"},
            ]
        )
        agent = ReActBaselineAgent(model)

        agent.handle("查询 O10086")

        observation = json.loads(agent.messages[3]["content"])
        self.assertEqual(observation["error"]["code"], "TOOL_ARGUMENT_MISMATCH")

    def test_max_steps_prevents_unbounded_tool_loop(self) -> None:
        model = ScriptedModelClient(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_1", "list_orders", "{}")],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_2", "list_orders", "{}")],
                },
            ]
        )
        agent = ReActBaselineAgent(model, max_steps=2)

        answer = agent.handle("一直查订单")

        self.assertIn("最多 2 次模型决策", answer)
        self.assertEqual(agent.trace[-1].event_type, "guard")
        self.assertEqual(agent.trace[-1].detail["reason"], "MAX_STEPS_EXCEEDED")


class DeepSeekClientTests(unittest.TestCase):
    def test_default_request_uses_v4_flash_tool_calls_and_non_thinking_mode(self) -> None:
        captured: dict = {}

        def fake_transport(url, payload, headers, timeout):
            captured.update(
                {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
            )
            return {
                "model": "deepseek-v4-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "你好"},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }

        client = DeepSeekClient(api_key="test-key", transport=fake_transport)
        completion = client.complete([{"role": "user", "content": "你好"}], BASELINE_TOOLS)

        self.assertEqual(captured["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured["payload"]["model"], "deepseek-v4-flash")
        self.assertEqual(captured["payload"]["thinking"], {"type": "disabled"})
        self.assertEqual(captured["payload"]["tool_choice"], "auto")
        self.assertEqual(completion.usage["prompt_tokens"], 10)

    def test_env_loader_fails_cleanly_without_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(DeepSeekConfigurationError):
                DeepSeekClient.from_env()


if __name__ == "__main__":
    unittest.main(verbosity=2)
