import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from agent import PolicyKnowledge, ToolGateway
from baseline.tool_catalog import SessionToolExecutor
from mcp_server.catalog import DEFAULT_CUSTOMER_SCOPES
from mcp_server.models import AuthContext, CircuitState
from mcp_server.server import MCPToolServer
from repositories import InMemoryStore


RETURN_ARGUMENTS = {
    "order_id": "O10086",
    "item_ids": ["I002"],
    "refund_amount": 129.0,
    "reason": "no_reason_return",
    "idempotency_key": "stage4-return-001",
    "user_confirmation": True,
}


class MCPReliabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = InMemoryStore()
        self.gateway = ToolGateway(self.store, PolicyKnowledge())
        self.servers: list[MCPToolServer] = []

    def tearDown(self) -> None:
        for server in self.servers:
            server.close()

    def executor(self, **server_kwargs) -> SessionToolExecutor:
        server = MCPToolServer(self.gateway, **server_kwargs)
        self.servers.append(server)
        return SessionToolExecutor(user_id="U001", mcp_server=server)

    def test_mcp_lists_schema_and_handles_jsonrpc_tool_call(self) -> None:
        executor = self.executor()
        tools = executor.client.list_tools()

        get_order = next(tool for tool in tools if tool["name"] == "get_order")
        self.assertEqual(get_order["inputSchema"]["required"], ["order_id"])
        self.assertFalse(get_order["annotations"]["destructiveHint"])

        response = executor.mcp_server.handle_jsonrpc(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "get_order", "arguments": {"order_id": "O10086"}},
            },
            executor.auth_context,
        )

        self.assertEqual(response["id"], 7)
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(
            response["result"]["structuredContent"]["data"]["order"]["order_id"],
            "O10086",
        )

    def test_schema_validation_rejects_bad_type_before_backend(self) -> None:
        executor = self.executor()

        result = executor.execute("get_order", {"order_id": 10086})

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "TOOL_ARGUMENT_MISMATCH")
        self.assertEqual(len(self.gateway.trace), 0)
        self.assertEqual(executor.audit_log[-1].decision, "rejected")
        self.assertEqual(executor.audit_log[-1].attempts, 0)

    def test_model_cannot_override_identity_and_attempt_is_audited(self) -> None:
        executor = self.executor()

        result = executor.execute(
            "get_order", {"order_id": "O20001", "user_id": "U002"}
        )

        self.assertEqual(result.error_code, "IDENTITY_OVERRIDE_ATTEMPT")
        self.assertEqual(executor.audit_log[-1].decision, "denied")
        self.assertEqual(len(self.gateway.trace), 0)

    def test_scope_permission_denies_write_before_execution(self) -> None:
        server = MCPToolServer(self.gateway)
        self.servers.append(server)
        executor = SessionToolExecutor(
            user_id="U001",
            mcp_server=server,
            scopes={"orders:read", "returns:quote"},
        )

        result = executor.execute("create_return_request", RETURN_ARGUMENTS)

        self.assertEqual(result.status, "FORBIDDEN")
        self.assertEqual(result.error_code, "INSUFFICIENT_SCOPE")
        self.assertEqual(len(self.store.return_requests), 0)
        self.assertNotIn(
            "create_return_request", {tool["name"] for tool in executor.client.list_tools()}
        )

    def test_read_timeout_is_retried_and_one_logical_call_is_audited(self) -> None:
        executor = self.executor(retry_backoff_seconds=0)
        self.gateway.inject_failure_once("get_order", "TIMEOUT")

        result = executor.execute("get_order", {"order_id": "O10086"})

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["attempts"], 2)
        self.assertEqual(executor.audit_log[-1].attempts, 2)
        self.assertEqual(
            [event.status for event in self.gateway.trace[-2:]], ["TIMEOUT", "OK"]
        )

    def test_real_execution_timeout_is_enforced(self) -> None:
        original = self.gateway.tools["get_product"].handler

        def slow_handler(product_id: str):
            time.sleep(0.05)
            return original(product_id)

        self.gateway.tools["get_product"].handler = slow_handler
        executor = self.executor(
            timeout_overrides={"get_product": 0.005},
            max_attempt_overrides={"get_product": 1},
            retry_backoff_seconds=0,
        )

        result = executor.execute("get_product", {"product_id": "P002"})

        self.assertEqual(result.status, "TIMEOUT")
        self.assertEqual(result.error_code, "TOOL_TIMEOUT")
        self.assertEqual(executor.audit_log[-1].attempts, 1)

    def test_circuit_breaker_opens_after_repeated_transient_failures(self) -> None:
        executor = self.executor(
            circuit_failure_threshold=2,
            circuit_recovery_seconds=60,
            max_attempt_overrides={"get_order": 1},
            retry_backoff_seconds=0,
        )
        for _ in range(2):
            self.gateway.inject_failure_once("get_order", "TIMEOUT")
            result = executor.execute("get_order", {"order_id": "O10086"})
            self.assertEqual(result.status, "TIMEOUT")

        blocked = executor.execute("get_order", {"order_id": "O10086"})

        self.assertEqual(blocked.status, "CIRCUIT_OPEN")
        self.assertEqual(
            executor.mcp_server.breakers["get_order"].state, CircuitState.OPEN
        )
        self.assertEqual(executor.audit_log[-1].attempts, 0)

    def test_circuit_breaker_half_open_probe_recovers(self) -> None:
        now = [0.0]
        executor = self.executor(
            circuit_failure_threshold=1,
            circuit_recovery_seconds=10,
            max_attempt_overrides={"get_order": 1},
            retry_backoff_seconds=0,
            clock=lambda: now[0],
        )
        self.gateway.inject_failure_once("get_order", "TIMEOUT")
        failed = executor.execute("get_order", {"order_id": "O10086"})
        now[0] = 5.0
        blocked = executor.execute("get_order", {"order_id": "O10086"})
        now[0] = 11.0
        recovered = executor.execute("get_order", {"order_id": "O10086"})

        self.assertEqual(failed.status, "TIMEOUT")
        self.assertEqual(blocked.status, "CIRCUIT_OPEN")
        self.assertTrue(recovered.ok)
        self.assertEqual(
            executor.mcp_server.breakers["get_order"].state, CircuitState.CLOSED
        )

    def test_write_timeout_is_not_retried_and_is_reconciled(self) -> None:
        executor = self.executor(retry_backoff_seconds=0)
        self.gateway.inject_failure_once(
            "create_return_request", "TIMEOUT_AFTER_COMMIT"
        )

        result = executor.execute("create_return_request", RETURN_ARGUMENTS)

        self.assertEqual(result.status, "UNKNOWN_COMMIT")
        self.assertEqual(result.metadata["attempts"], 1)
        self.assertEqual(result.metadata["reconciliation_tool"], "get_return_status")
        self.assertEqual(len(self.store.return_requests), 1)

        reconciled = executor.execute(
            "get_return_status",
            {"idempotency_key": RETURN_ARGUMENTS["idempotency_key"]},
        )
        self.assertTrue(reconciled.ok)
        self.assertTrue(reconciled.data["found"])
        self.assertEqual(len(self.store.return_requests), 1)

    def test_idempotency_key_cannot_be_reused_for_different_payload(self) -> None:
        executor = self.executor()
        first = executor.execute("create_return_request", RETURN_ARGUMENTS)
        changed = {**RETURN_ARGUMENTS, "refund_amount": 128.0}

        second = executor.execute("create_return_request", changed)

        self.assertTrue(first.ok)
        self.assertEqual(second.status, "CONFLICT")
        self.assertEqual(second.error_code, "IDEMPOTENCY_KEY_REUSE")
        self.assertEqual(len(self.store.return_requests), 1)

    def test_repository_idempotency_also_blocks_cross_user_key_reuse(self) -> None:
        executor = self.executor()
        first = executor.execute("create_return_request", RETURN_ARGUMENTS)
        other_server = MCPToolServer(self.gateway)
        self.servers.append(other_server)
        other = SessionToolExecutor(
            user_id="U002", mcp_server=other_server, scopes=DEFAULT_CUSTOMER_SCOPES
        )
        changed_user_operation = {
            **RETURN_ARGUMENTS,
            "order_id": "O20001",
            "item_ids": ["I004"],
            "refund_amount": 499.0,
        }

        second = other.execute("create_return_request", changed_user_operation)

        self.assertTrue(first.ok)
        self.assertEqual(second.error_code, "IDEMPOTENCY_KEY_REUSE")
        self.assertEqual(len(self.store.return_requests), 1)

    def test_concurrent_same_key_write_creates_one_business_record(self) -> None:
        executor = self.executor()

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda _: executor.execute(
                        "create_return_request", dict(RETURN_ARGUMENTS)
                    ),
                    range(4),
                )
            )

        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(len(self.store.return_requests), 1)
        request_ids = {result.data["request_id"] for result in results}
        self.assertEqual(len(request_ids), 1)

    def test_ticket_write_is_idempotent_and_audit_redacts_description(self) -> None:
        executor = self.executor()
        arguments = {
            "subject": "售后人工处理",
            "description": "用户手机号 13800000000，需要人工核查",
            "order_id": "O10086",
            "idempotency_key": "stage4-ticket-001",
        }

        first = executor.execute("create_ticket", arguments)
        second = executor.execute("create_ticket", arguments)

        self.assertTrue(first.ok)
        self.assertTrue(second.ok)
        self.assertEqual(len(self.store.tickets), 1)
        audit = executor.audit_log[-1]
        self.assertEqual(audit.sanitized_arguments["description"], "<redacted>")
        self.assertTrue(
            audit.sanitized_arguments["idempotency_key"].startswith("<sha256:")
        )
        self.assertIsNotNone(audit.idempotency_key_digest)

    def test_ticket_unknown_commit_can_be_reconciled_without_duplicate(self) -> None:
        executor = self.executor()
        arguments = {
            "subject": "售后人工处理",
            "description": "自动处理状态未知",
            "order_id": "O10086",
            "idempotency_key": "stage4-ticket-unknown-001",
        }
        self.gateway.inject_failure_once("create_ticket", "TIMEOUT_AFTER_COMMIT")

        result = executor.execute("create_ticket", arguments)
        status = executor.execute(
            "get_ticket_status",
            {"idempotency_key": arguments["idempotency_key"]},
        )

        self.assertEqual(result.status, "UNKNOWN_COMMIT")
        self.assertEqual(result.metadata["reconciliation_tool"], "get_ticket_status")
        self.assertTrue(status.ok)
        self.assertTrue(status.data["found"])
        self.assertEqual(len(self.store.tickets), 1)

    def test_invalid_tool_response_is_retried_then_rejected(self) -> None:
        self.gateway.tools["get_product"].handler = lambda product_id: {"wrong": True}
        executor = self.executor(retry_backoff_seconds=0)

        result = executor.execute("get_product", {"product_id": "P002"})

        self.assertEqual(result.status, "INVALID_RESPONSE")
        self.assertEqual(result.error_code, "INVALID_TOOL_RESPONSE")
        self.assertEqual(result.metadata["attempts"], 2)
        self.assertEqual(executor.audit_log[-1].attempts, 2)


class MCPProtocolPermissionTests(unittest.TestCase):
    def test_tools_list_only_exposes_authorized_tools(self) -> None:
        gateway = ToolGateway(InMemoryStore(), PolicyKnowledge())
        server = MCPToolServer(gateway)
        self.addCleanup(server.close)
        context = AuthContext(
            user_id="U001",
            session_id="read-only-session",
            scopes=frozenset({"orders:read"}),
        )

        response = server.handle_jsonrpc(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            context,
        )

        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertEqual(names, {"list_orders", "get_order"})
        self.assertNotEqual(context.scopes, DEFAULT_CUSTOMER_SCOPES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
