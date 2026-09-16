"""Common expert boundary for MCP calls and structured observations."""

from __future__ import annotations

from typing import Callable, Protocol

from agent import ToolResult
from routing.models import ExpertName, ExpertResult, ExpertStatus
from structured.models import StructuredTaskState, ToolFact


ToolCaller = Callable[[str, dict], ToolResult]


class Expert(Protocol):
    name: ExpertName

    def execute(self, state: StructuredTaskState) -> ExpertResult:
        ...


class BaseExpert:
    name: ExpertName
    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, tool_caller: ToolCaller) -> None:
        self.call_tool = tool_caller

    def observe(
        self, state: StructuredTaskState, tool_name: str, arguments: dict
    ) -> ToolResult:
        if tool_name not in self.allowed_tools:
            return ToolResult(
                False,
                "FORBIDDEN",
                error_code="EXPERT_TOOL_FORBIDDEN",
                message=f"专家 {self.name.value} 无权调用工具 {tool_name}。",
            )
        result = self.call_tool(tool_name, arguments)
        state.facts[f"{self.name.value}:{tool_name}:{len(state.facts) + 1}"] = ToolFact(
            key=tool_name,
            value=result.data if result.ok else {"error_code": result.error_code},
            tool=tool_name,
            observed_turn=state.turn_index,
        )
        return result

    def escalate(self, reason: str, result: ToolResult | None = None) -> ExpertResult:
        suffix = ""
        if result is not None:
            suffix = f"：{result.error_code or result.status}"
        return ExpertResult(
            expert=self.name,
            status=ExpertStatus.ESCALATE,
            reason_code="TOOL_OR_SAFETY_FAILURE",
            escalation_reason=reason + suffix,
        )
