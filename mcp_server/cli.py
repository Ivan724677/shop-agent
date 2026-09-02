"""JSON-lines stdio transport for the stage-four MCP server."""

from __future__ import annotations

import argparse
import json
import sys
import uuid

from agent import PolicyKnowledge, ToolGateway
from repositories import InMemoryStore

from .catalog import DEFAULT_CUSTOMER_SCOPES
from .models import AuthContext
from .server import MCPToolServer


def main() -> int:
    parser = argparse.ArgumentParser(description="售后工具 MCP stdio server")
    parser.add_argument("--user-id", default="U001")
    args = parser.parse_args()
    gateway = ToolGateway(InMemoryStore(), PolicyKnowledge())
    server = MCPToolServer(gateway)
    context = AuthContext(
        user_id=args.user_id,
        session_id=str(uuid.uuid4()),
        scopes=DEFAULT_CUSTOMER_SCOPES,
    )
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                response = server.handle_jsonrpc(message, context)
            except json.JSONDecodeError:
                response = server._rpc_error(None, -32700, "Parse error")
            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
