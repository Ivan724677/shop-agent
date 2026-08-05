import unittest
from pathlib import Path

from agent import CustomerServiceAgent, PolicyKnowledge
from domain.models import (
    OrderStatus,
    RefundStatus,
    ShipmentStatus,
    TicketStatus,
)
from domain.state_machine import InvalidTransition
from repositories import InMemoryStore
from scenarios.loader import load_corpus


class StageOneDomainTests(unittest.TestCase):
    def test_seed_contains_independent_domain_entities(self) -> None:
        store = InMemoryStore()

        self.assertIn("U001", store.users)
        self.assertIn("P002", store.products)
        self.assertIn("O10086", store.orders)
        self.assertIn("SHP10086A", store.shipments)
        self.assertEqual(store.orders["O10086"].items[1].product_id, "P002")
        self.assertEqual(store.orders["O10086"].items[1].shipment_id, "SHP10086B")

    def test_order_and_shipment_transitions_are_guarded(self) -> None:
        store = InMemoryStore()

        store.transition_order("O10086", OrderStatus.DELIVERED, "fulfillment_service")
        self.assertEqual(store.orders["O10086"].status, OrderStatus.DELIVERED)
        self.assertEqual(len(store.orders["O10086"].status_history), 1)

        with self.assertRaises(InvalidTransition):
            store.transition_order("O10086", OrderStatus.PAID, "bad_actor")

        store.transition_shipment("SHP10086A", ShipmentStatus.OUT_FOR_DELIVERY, "logistics_service")
        self.assertEqual(store.shipments["SHP10086A"].status, ShipmentStatus.OUT_FOR_DELIVERY)

    def test_return_and_refund_are_separate_records(self) -> None:
        agent = CustomerServiceAgent()
        agent.handle("我要退订单 O10086 里的鞋")
        agent.handle("确认退货")

        request = next(iter(agent.store.return_requests.values()))
        refund = next(iter(agent.store.refund_transactions.values()))
        self.assertEqual(request.request_id, refund.request_id)
        self.assertEqual(refund.amount, request.refund_amount)
        self.assertEqual(refund.status, RefundStatus.PENDING)
        self.assertNotEqual(request.request_id, refund.refund_id)

    def test_ticket_has_its_own_lifecycle(self) -> None:
        store = InMemoryStore()
        ticket = store.create_ticket("U001", "物流异常", "包裹轨迹两天没有更新")
        store.transition_ticket(ticket.ticket_id, TicketStatus.ASSIGNED, "dispatcher", "分配物流组")
        store.transition_ticket(ticket.ticket_id, TicketStatus.IN_PROGRESS, "logistics_team")
        store.transition_ticket(ticket.ticket_id, TicketStatus.RESOLVED, "logistics_team", "已联系承运商")
        store.transition_ticket(ticket.ticket_id, TicketStatus.CLOSED, "system", "用户确认")

        self.assertEqual(ticket.status, TicketStatus.CLOSED)
        self.assertEqual([entry.to_status for entry in ticket.timeline], [
            "open",
            "assigned",
            "in_progress",
            "resolved",
            "closed",
        ])

        with self.assertRaises(InvalidTransition):
            store.transition_ticket(ticket.ticket_id, TicketStatus.IN_PROGRESS, "bad_actor")

    def test_new_business_tools_expose_user_product_and_logistics(self) -> None:
        agent = CustomerServiceAgent()

        user = agent.gateway.call("get_user", user_id="U001")
        product = agent.gateway.call("get_product", product_id="P002")
        shipments = agent.gateway.call("list_shipments", user_id="U001", order_id="O10086")
        ticket = agent.gateway.call(
            "create_ticket",
            user_id="U001",
            subject="物流异常",
            description="请人工核实包裹轨迹",
        )

        self.assertTrue(user.ok)
        self.assertEqual(user.data["user"]["membership_level"], "gold")
        self.assertTrue(product.ok)
        self.assertEqual(product.data["product"]["product_id"], "P002")
        self.assertEqual(len(shipments.data["shipments"]), 2)
        self.assertTrue(ticket.ok)
        self.assertEqual(ticket.data["ticket"]["status"], "open")


class ScenarioCorpusTests(unittest.TestCase):
    def test_corpus_is_loadable_and_has_expected_layers(self) -> None:
        scenarios = load_corpus(Path("scenarios"))
        self.assertGreaterEqual(len(scenarios), 130)
        self.assertTrue(any("golden" in scenario.tags for scenario in scenarios))
        self.assertTrue(any("fault" in scenario.tags for scenario in scenarios))
        self.assertTrue(any("generated" in scenario.tags for scenario in scenarios))
        self.assertEqual(
            len({scenario.scenario_id for scenario in scenarios}), len(scenarios)
        )

    def test_policy_documents_are_loaded_from_seed_data(self) -> None:
        knowledge = PolicyKnowledge()
        document_ids = {document["id"] for document in knowledge.documents}
        self.assertEqual(
            document_ids,
            {"policy_standard_v1", "policy_custom_v1", "policy_quality_v1"},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
