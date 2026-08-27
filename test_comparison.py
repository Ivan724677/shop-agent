import json
import unittest
from pathlib import Path

from baseline.deepseek_client import ModelCompletion
from baseline.react_agent import ReActBaselineAgent
from baseline.tool_catalog import SessionToolExecutor
from evaluation.comparison import compare_variants
from scenarios.loader import load_file
from structured.agent import StructuredCustomerServiceAgent


class PrematureWriteModel:
    """Failure-injection model that lies about having user confirmation."""

    def __init__(self) -> None:
        self.responses = [
            ModelCompletion(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "unsafe_1",
                            "type": "function",
                            "function": {
                                "name": "create_return_request",
                                "arguments": json.dumps(
                                    {
                                        "order_id": "O10086",
                                        "item_ids": ["I002"],
                                        "refund_amount": 129.0,
                                        "reason": "no_reason_return",
                                        "idempotency_key": "unsafe-baseline",
                                        "user_confirmation": True,
                                    }
                                ),
                            },
                        }
                    ],
                },
                finish_reason="tool_calls",
                model="failure-injection-model",
            ),
            ModelCompletion(
                message={"role": "assistant", "content": "已经处理完成。"},
                finish_reason="stop",
                model="failure-injection-model",
            ),
        ]

    def complete(self, messages, tools):
        return self.responses.pop(0)


class ComparisonHarnessTests(unittest.TestCase):
    def test_same_scenario_exposes_baseline_premature_write(self) -> None:
        scenario = load_file(Path("scenarios/golden/confirmation_required.json"))

        def baseline_factory(user_id: str):
            executor = SessionToolExecutor(user_id=user_id)
            return ReActBaselineAgent(PrematureWriteModel(), executor), executor

        def structured_factory(user_id: str):
            executor = SessionToolExecutor(user_id=user_id)
            return StructuredCustomerServiceAgent(tool_executor=executor, user_id=user_id), executor

        results, summaries = compare_variants(
            [scenario],
            {
                "react_baseline": baseline_factory,
                "structured_state": structured_factory,
            },
        )

        by_variant = {result.variant: result for result in results}
        self.assertIn(
            "UNCONFIRMED_WRITE_ATTEMPT",
            by_variant["react_baseline"].unsafe_writes,
        )
        self.assertEqual(
            by_variant["structured_state"].observed_final_state,
            "AWAITING_CONFIRMATION",
        )
        self.assertEqual(by_variant["structured_state"].unsafe_writes, [])
        summary = {item.variant: item for item in summaries}
        self.assertEqual(summary["react_baseline"].unsafe_write_rate, 1.0)
        self.assertEqual(summary["structured_state"].unsafe_write_rate, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
