"""In-process MCP client used by Agents while preserving the protocol boundary."""

from __future__ import annotations

import uuid
from typing import Any

from agent import ToolResult

from .models import AuthContext
from .server import MCPToolServer


class InProcessMCPClient:
    def __init__(self, server: MCPToolServer, context: AuthContext) -> None:
        self.server = server
        self.context = context

    def list_tools(self) -> list[dict[str, Any]]:
        return self.server.list_tools(self.context)

    def call_tool(
        self,
        tool_name: str,
        arguments: Any,
        *,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> ToolResult:
        return self.server.call_tool(
            tool_name,
            arguments,
            self.context,
            request_id=request_id or str(uuid.uuid4()),
            correlation_id=correlation_id or self.context.session_id,
        )
