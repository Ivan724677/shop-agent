import unittest

from agent import CustomerServiceAgent


class CustomerServiceAgentTests(unittest.TestCase):
    def test_exclusion_scope_does_not_cross_product_boundary(self) -> None:
        agent = CustomerServiceAgent()

        agent.handle("我要退订单 O10086 里的鞋，耳机不退")

        self.assertEqual(agent.state.selected_item_ids, {"I002"})
        self.assertEqual(agent.state.excluded_item_ids, {"I001"})

    def test_partial_return_needs_confirmation_and_only_selects_shoes(self) -> None:
        agent = CustomerServiceAgent()

        reply = agent.handle("我要退订单 O10086 里的鞋，耳机不退")

        self.assertIn("确认退货", reply)
        self.assertEqual(agent.state.selected_item_ids, {"I002"})
        self.assertEqual(agent.state.refund_amount, 129.0)

        reply = agent.handle("确认退货")

        self.assertIn("已创建", reply)
        request = next(iter(agent.store.return_requests.values()))
        self.assertEqual(request["item_ids"], ["I002"])
        self.assertEqual(request["refund_amount"], 129.0)

    def test_timeout_after_commit_is_reconciled_without_duplicate_write(self) -> None:
        agent = CustomerServiceAgent()
        agent.gateway.inject_failure_once("create_return_request", "TIMEOUT_AFTER_COMMIT")

        agent.handle("我要退订单 O10086 里的鞋")
        reply = agent.handle("确认退货")

        self.assertIn("幂等键对账确认", reply)
        self.assertEqual(len(agent.store.return_requests), 1)
        self.assertTrue(any(event.status == "UNKNOWN_COMMIT" for event in agent.gateway.trace))
        self.assertTrue(any(alert["type"] == "unknown_commit" for alert in agent.monitor_alerts()))

    def test_custom_product_is_blocked_by_policy(self) -> None:
        agent = CustomerServiceAgent()

        reply = agent.handle("我要退订单 O10087 里的马克杯")

        self.assertIn("定制商品", reply)
        self.assertEqual(len(agent.store.return_requests), 0)

    def test_query_without_order_id_lists_orders_before_reading_details(self) -> None:
        agent = CustomerServiceAgent()

        reply = agent.handle("帮我查询订单")

        self.assertIn("O10086", reply)
        self.assertIn("O10087", reply)


if __name__ == "__main__":
    unittest.main(verbosity=2)
