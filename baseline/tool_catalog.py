"""OpenAI-compatible tool schemas and a session-bound execution adapter."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import uuid
from typing import Any

from agent import PolicyKnowledge, ToolGateway, ToolResult
from mcp_server.catalog import DEFAULT_CUSTOMER_SCOPES, OPENAI_TOOLS
from mcp_server.client import InProcessMCPClient
from mcp_server.models import AuthContext
from mcp_server.server import MCPToolServer
from repositories import InMemoryStore

BASELINE_TOOLS = OPENAI_TOOLS


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def tool_result_as_json(result: ToolResult) -> str:
    payload = {
        "ok": result.ok,
        "status": result.status,
        "data": _jsonable(result.data),
        "error": (
            {
                "code": result.error_code,
                "message": result.message,
                "retryable": result.retryable,
            }
            if not result.ok
            else None
        ),
        "metadata": _jsonable(result.metadata),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class SessionToolExecutor:
    """Bind authenticated user identity outside model-controlled arguments."""

    def __init__(
        self,
        user_id: str = "U001",
        store: InMemoryStore | None = None,
        gateway: ToolGateway | None = None,
        mcp_server: MCPToolServer | None = None,
        scopes: frozenset[str] | set[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        self.user_id = user_id
        if mcp_server is not None:
            self.gateway = mcp_server.gateway
            self.store = store or self.gateway.store
            self.mcp_server = mcp_server
        elif gateway is not None:
            self.gateway = gateway
            self.store = store or gateway.store
            self.mcp_server = MCPToolServer(self.gateway)
        else:
            self.store = store or InMemoryStore()
            self.gateway = ToolGateway(self.store, PolicyKnowledge())
            self.mcp_server = MCPToolServer(self.gateway)
        self.auth_context = AuthContext(
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
            scopes=frozenset(scopes) if scopes is not None else DEFAULT_CUSTOMER_SCOPES,
        )
        self.client = InProcessMCPClient(self.mcp_server, self.auth_context)
        self.audit_log = self.mcp_server.audit_log
        self.allowed_tool_names = {
            tool["function"]["name"] for tool in BASELINE_TOOLS
        }

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        return self.client.call_tool(tool_name, arguments)

    def audit_as_json(self) -> str:
        return json.dumps(
            [_jsonable(event) for event in self.audit_log],
            ensure_ascii=False,
            indent=2,
        )
