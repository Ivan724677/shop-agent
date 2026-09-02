"""Transport-neutral MCP server with a deterministic reliability pipeline."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from threading import Lock
from typing import Any, Callable

from agent import ToolGateway, ToolResult

from .catalog import MCP_TOOLS, ToolDefinition
from .models import AuditEvent, AuthContext, CircuitState
from .validation import SchemaValidationError, validate_json


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_seconds: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold <= 0 or recovery_seconds < 0:
            raise ValueError("熔断参数不合法。")
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.clock = clock
        self.state = CircuitState.CLOSED
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.HALF_OPEN:
                return False
            if (
                self.opened_at is not None
                and self.clock() - self.opened_at >= self.recovery_seconds
            ):
                self.state = CircuitState.HALF_OPEN
                return True
            return False

    def success(self) -> None:
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failures = 0
            self.opened_at = None

    def transient_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == CircuitState.HALF_OPEN or self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = self.clock()


class MCPToolServer:
    """Expose tools/list and tools/call while enforcing reliability invariants."""

    PROTOCOL_VERSION = "2025-06-18"

    def __init__(
        self,
        gateway: ToolGateway,
        *,
        tool_definitions: dict[str, ToolDefinition] | None = None,
        circuit_failure_threshold: int = 3,
        circuit_recovery_seconds: float = 30.0,
        timeout_overrides: dict[str, float] | None = None,
        max_attempt_overrides: dict[str, int] | None = None,
        retry_backoff_seconds: float = 0.01,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.gateway = gateway
        self.tool_definitions = tool_definitions or MCP_TOOLS
        self.timeout_overrides = timeout_overrides or {}
        self.max_attempt_overrides = max_attempt_overrides or {}
        self.retry_backoff_seconds = retry_backoff_seconds
        self.clock = clock
        self.audit_log: list[AuditEvent] = []
        self.idempotency_fingerprints: dict[tuple[str, str, str], str] = {}
        self._idempotency_lock = Lock()
        self._audit_lock = Lock()
        self.breakers = {
            name: CircuitBreaker(
                circuit_failure_threshold, circuit_recovery_seconds, clock
            )
            for name in self.tool_definitions
        }
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="mcp-tool")

    def list_tools(self, context: AuthContext) -> list[dict[str, Any]]:
        return [
            definition.as_mcp_tool()
            for definition in self.tool_definitions.values()
            if definition.required_scopes.issubset(context.scopes)
        ]

    def call_tool(
        self,
        tool_name: str,
        arguments: Any,
        context: AuthContext,
        *,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> ToolResult:
        started = self.clock()
        request_id = request_id or str(uuid.uuid4())
        correlation_id = correlation_id or context.session_id
        definition = self.tool_definitions.get(tool_name)
        if definition is None:
            result = ToolResult(
                False, "ERROR", error_code="UNKNOWN_TOOL", message=f"工具 {tool_name} 不存在。"
            )
            self._audit_unknown(
                tool_name, arguments, context, request_id, correlation_id, started, result
            )
            return result

        breaker = self.breakers[tool_name]
        circuit_before = breaker.state.value
        early = self._validate_request(definition, arguments, context)
        if early is not None:
            self._finish(
                definition,
                arguments,
                context,
                request_id,
                correlation_id,
                started,
                early,
                attempts=0,
                circuit_before=circuit_before,
                decision="denied" if early.status == "FORBIDDEN" else "rejected",
            )
            return early

        if not breaker.allow():
            result = ToolResult(
                False,
                "CIRCUIT_OPEN",
                error_code="CIRCUIT_OPEN",
                message="工具暂时熔断，请稍后重试或转人工处理。",
                retryable=True,
            )
            self._finish(
                definition,
                arguments,
                context,
                request_id,
                correlation_id,
                started,
                result,
                attempts=0,
                circuit_before=circuit_before,
                decision="blocked",
            )
            return result

        idempotency_error = self._reserve_idempotency(
            definition, arguments, context
        )
        if idempotency_error is not None:
            self._finish(
                definition,
                arguments,
                context,
                request_id,
                correlation_id,
                started,
                idempotency_error,
                attempts=0,
                circuit_before=circuit_before,
                decision="rejected",
            )
            return idempotency_error

        safe_arguments = dict(arguments)
        if definition.inject_user_id:
            safe_arguments["user_id"] = context.user_id
        max_attempts = self.max_attempt_overrides.get(
            tool_name, definition.max_attempts
        )
        timeout = self.timeout_overrides.get(
            tool_name, definition.timeout_seconds
        )
        max_attempts = max(1, max_attempts)
        attempts = 0
        result: ToolResult
        while attempts < max_attempts:
            attempts += 1
            result = self._invoke_once(definition, safe_arguments, timeout)
            if result.ok:
                try:
                    validate_json(result.data, definition.output_schema)
                except SchemaValidationError as exc:
                    result = ToolResult(
                        False,
                        "INVALID_RESPONSE",
                        error_code="INVALID_TOOL_RESPONSE",
                        message=f"工具返回不符合 output schema：{exc}",
                        retryable=not definition.side_effect,
                    )
            if not self._should_retry(definition, result, attempts, max_attempts):
                break
            if self.retry_backoff_seconds > 0:
                time.sleep(self.retry_backoff_seconds * (2 ** (attempts - 1)))

        if result.ok:
            breaker.success()
        elif self._is_transient(result):
            breaker.transient_failure()
        result.metadata.update(
            {
                "request_id": request_id,
                "correlation_id": correlation_id,
                "attempts": attempts,
                "reconciliation_tool": (
                    definition.reconciliation_tool
                    if result.status == "UNKNOWN_COMMIT"
                    else None
                ),
            }
        )
        self._finish(
            definition,
            arguments,
            context,
            request_id,
            correlation_id,
            started,
            result,
            attempts=attempts,
            circuit_before=circuit_before,
            decision="executed",
        )
        return result

    def handle_jsonrpc(self, message: dict[str, Any], context: AuthContext) -> dict[str, Any] | None:
        request_id = message.get("id")
        method = message.get("method")
        if method == "notifications/initialized":
            return None
        if message.get("jsonrpc") != "2.0" or not isinstance(method, str):
            return self._rpc_error(request_id, -32600, "Invalid Request")
        if method == "initialize":
            return self._rpc_result(
                request_id,
                {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "after-sales-reliability", "version": "0.4.0"},
                },
            )
        if method == "ping":
            return self._rpc_result(request_id, {})
        if method == "tools/list":
            return self._rpc_result(request_id, {"tools": self.list_tools(context)})
        if method == "tools/call":
            params = message.get("params") or {}
            if not isinstance(params, dict) or not isinstance(params.get("name"), str):
                return self._rpc_error(request_id, -32602, "Invalid tools/call params")
            result = self.call_tool(
                params["name"],
                params.get("arguments", {}),
                context,
                request_id=str(request_id),
                correlation_id=context.session_id,
            )
            structured = _result_payload(result)
            return self._rpc_result(
                request_id,
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(structured, ensure_ascii=False),
                        }
                    ],
                    "structuredContent": structured,
                    "isError": not result.ok,
                },
            )
        return self._rpc_error(request_id, -32601, "Method not found")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _validate_request(
        self, definition: ToolDefinition, arguments: Any, context: AuthContext
    ) -> ToolResult | None:
        if not isinstance(arguments, dict):
            return ToolResult(
                False,
                "VALIDATION_ERROR",
                error_code="ARGUMENTS_NOT_OBJECT",
                message="工具参数必须是 JSON 对象。",
            )
        if "user_id" in arguments:
            return ToolResult(
                False,
                "FORBIDDEN",
                error_code="IDENTITY_OVERRIDE_ATTEMPT",
                message="调用方不能覆盖已认证会话中的用户身份。",
            )
        missing_scopes = sorted(definition.required_scopes - context.scopes)
        if missing_scopes:
            return ToolResult(
                False,
                "FORBIDDEN",
                error_code="INSUFFICIENT_SCOPE",
                message="缺少工具权限：" + ", ".join(missing_scopes),
            )
        try:
            validate_json(arguments, definition.input_schema)
        except SchemaValidationError as exc:
            return ToolResult(
                False,
                "VALIDATION_ERROR",
                error_code="TOOL_ARGUMENT_MISMATCH",
                message=f"工具参数不符合 input schema：{exc}",
            )
        if (
            definition.name == "create_return_request"
            and arguments.get("user_confirmation") is not True
        ):
            return ToolResult(
                False,
                "ERROR",
                error_code="CONFIRMATION_REQUIRED",
                message="高风险操作需要用户明确确认。",
            )
        return None

    def _reserve_idempotency(
        self,
        definition: ToolDefinition,
        arguments: dict[str, Any],
        context: AuthContext,
    ) -> ToolResult | None:
        field = definition.idempotency_field
        if field is None:
            return None
        key = str(arguments[field])
        registry_key = (context.user_id, definition.name, key)
        digest = _digest(arguments)
        with self._idempotency_lock:
            existing = self.idempotency_fingerprints.get(registry_key)
            if existing is not None and existing != digest:
                return ToolResult(
                    False,
                    "CONFLICT",
                    error_code="IDEMPOTENCY_KEY_REUSE",
                    message="同一幂等键不能用于不同的操作参数。",
                )
            self.idempotency_fingerprints[registry_key] = digest
        return None

    def _invoke_once(
        self, definition: ToolDefinition, arguments: dict[str, Any], timeout: float
    ) -> ToolResult:
        future = self._pool.submit(self.gateway.call, definition.name, **arguments)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            if definition.side_effect:
                return ToolResult(
                    False,
                    "UNKNOWN_COMMIT",
                    error_code="UNKNOWN_COMMIT",
                    message="写工具超时，提交结果未知；禁止直接重试，必须先对账。",
                )
            return ToolResult(
                False,
                "TIMEOUT",
                error_code="TOOL_TIMEOUT",
                message="只读工具调用超时。",
                retryable=True,
            )
        except Exception as exc:  # The boundary must turn handler failures into observations.
            return ToolResult(
                False,
                "ERROR",
                error_code="TOOL_EXECUTION_ERROR",
                message=f"工具执行异常：{type(exc).__name__}",
            )

    @staticmethod
    def _should_retry(
        definition: ToolDefinition,
        result: ToolResult,
        attempts: int,
        max_attempts: int,
    ) -> bool:
        return (
            not definition.side_effect
            and result.retryable
            and result.status != "UNKNOWN_COMMIT"
            and attempts < max_attempts
        )

    @staticmethod
    def _is_transient(result: ToolResult) -> bool:
        return result.retryable or result.status in {
            "TIMEOUT",
            "UNKNOWN_COMMIT",
            "INVALID_RESPONSE",
        }

    def _finish(
        self,
        definition: ToolDefinition,
        arguments: Any,
        context: AuthContext,
        request_id: str,
        correlation_id: str,
        started: float,
        result: ToolResult,
        *,
        attempts: int,
        circuit_before: str,
        decision: str,
    ) -> None:
        audit = AuditEvent(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            correlation_id=correlation_id,
            request_id=request_id,
            actor_id=context.user_id,
            actor_type=context.actor_type,
            tool=definition.name,
            risk_level=definition.risk_level,
            side_effect=definition.side_effect,
            decision=decision,
            status=result.status,
            error_code=result.error_code,
            attempts=attempts,
            duration_ms=round((self.clock() - started) * 1000, 3),
            circuit_before=circuit_before,
            circuit_after=self.breakers[definition.name].state.value,
            argument_digest=_digest(arguments),
            sanitized_arguments=_sanitize(arguments, definition.input_schema),
            idempotency_key_digest=(
                _digest(arguments.get(definition.idempotency_field))
                if isinstance(arguments, dict)
                and definition.idempotency_field
                and definition.idempotency_field in arguments
                else None
            ),
        )
        with self._audit_lock:
            self.audit_log.append(audit)
        result.metadata.setdefault("audit_id", audit.audit_id)

    def _audit_unknown(
        self,
        tool_name: str,
        arguments: Any,
        context: AuthContext,
        request_id: str,
        correlation_id: str,
        started: float,
        result: ToolResult,
    ) -> None:
        audit = AuditEvent(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            correlation_id=correlation_id,
            request_id=request_id,
            actor_id=context.user_id,
            actor_type=context.actor_type,
            tool=tool_name,
            risk_level="unknown",
            side_effect=False,
            decision="rejected",
            status=result.status,
            error_code=result.error_code,
            attempts=0,
            duration_ms=round((self.clock() - started) * 1000, 3),
            circuit_before="unknown",
            circuit_after="unknown",
            argument_digest=_digest(arguments),
            sanitized_arguments=_sanitize(arguments, {}),
        )
        with self._audit_lock:
            self.audit_log.append(audit)
        result.metadata["audit_id"] = audit.audit_id

    @staticmethod
    def _rpc_result(request_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def _digest(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sanitize(value: Any, schema: dict[str, Any]) -> Any:
    if not isinstance(value, dict):
        return {"value_type": type(value).__name__}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    sensitive_names = {"phone", "email", "address", "description", "token", "api_key"}
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        item_schema = properties.get(key, {})
        if item_schema.get("x-sensitive") or key.lower() in sensitive_names:
            sanitized[key] = "<redacted>"
        elif key == "idempotency_key":
            sanitized[key] = "<sha256:" + _digest(item)[:12] + ">"
        else:
            sanitized[key] = _jsonable(item)
    return sanitized


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _result_payload(result: ToolResult) -> dict[str, Any]:
    return {
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
