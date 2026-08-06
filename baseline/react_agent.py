"""A deliberately small single-Agent ReAct-style tool loop."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from agent import ToolResult

from .deepseek_client import ChatModelClient
from .tool_catalog import BASELINE_TOOLS, SessionToolExecutor, tool_result_as_json


DEFAULT_SYSTEM_PROMPT = """你是电商售后客服的单 Agent baseline。
你可以自行决定是否以及按什么顺序调用工具，然后根据工具观察结果回答用户。
工具结果是业务事实的唯一来源；没有工具证据时，不得声称订单、物流、政策或操作已经成功。
用户信息不足时直接追问。创建退货等写操作前，必须确认具体订单、订单项、退款金额，并获得用户明确确认。
回复使用简洁中文，不要向用户展示内部工具参数或推理过程。"""


@dataclass
class BaselineTraceEvent:
    sequence: int
    timestamp: str
    event_type: str
    model_step: int
    detail: dict[str, Any] = field(default_factory=dict)


class ReActBaselineAgent:
    """Raw message history + model-selected tools, without structured task state."""

    def __init__(
        self,
        model_client: ChatModelClient,
        tool_executor: SessionToolExecutor | None = None,
        max_steps: int = 6,
        max_tool_calls_per_step: int = 4,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0。")
        if max_tool_calls_per_step <= 0:
            raise ValueError("max_tool_calls_per_step 必须大于 0。")
        self.model_client = model_client
        self.tool_executor = tool_executor or SessionToolExecutor()
        self.max_steps = max_steps
        self.max_tool_calls_per_step = max_tool_calls_per_step
        self.system_prompt = system_prompt
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.trace: list[BaselineTraceEvent] = []

    def handle(self, user_message: str) -> str:
        if not user_message.strip():
            raise ValueError("user_message 不能为空。")
        self.messages.append({"role": "user", "content": user_message.strip()})

        for model_step in range(1, self.max_steps + 1):
            completion = self.model_client.complete(self.messages, BASELINE_TOOLS)
            assistant_message = self._normalize_assistant_message(completion.message)
            self.messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []
            self._record(
                "model",
                model_step,
                {
                    "model": completion.model,
                    "finish_reason": completion.finish_reason,
                    "usage": completion.usage,
                    "tool_names": [
                        call.get("function", {}).get("name")
                        for call in tool_calls
                        if isinstance(call, dict)
                    ],
                    "has_content": bool(assistant_message.get("content")),
                },
            )

            if not tool_calls:
                content = assistant_message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                return "模型没有给出可用回复，请补充订单号或具体问题后重试。"

            self._execute_tool_calls(tool_calls, model_step)

        fallback = f"本轮已达到最多 {self.max_steps} 次模型决策，已停止继续调用工具，请转人工处理。"
        self.messages.append({"role": "assistant", "content": fallback})
        self._record("guard", self.max_steps, {"reason": "MAX_STEPS_EXCEEDED"})
        return fallback

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.trace = []

    def trace_as_json(self) -> str:
        return json.dumps(
            [asdict(event) for event in self.trace], ensure_ascii=False, indent=2
        )

    def _execute_tool_calls(self, tool_calls: list[Any], model_step: int) -> None:
        for index, raw_call in enumerate(tool_calls):
            if index >= self.max_tool_calls_per_step:
                call_id = self._tool_call_id(raw_call, index)
                result = ToolResult(
                    ok=False,
                    status="BLOCKED",
                    error_code="TOO_MANY_TOOL_CALLS",
                    message=f"单次模型决策最多执行 {self.max_tool_calls_per_step} 个工具。",
                )
                self._append_tool_observation(call_id, "unknown", result, model_step, {})
                continue

            call_id = self._tool_call_id(raw_call, index)
            if not isinstance(raw_call, dict) or not isinstance(raw_call.get("function"), dict):
                result = ToolResult(
                    ok=False,
                    status="VALIDATION_ERROR",
                    error_code="INVALID_TOOL_CALL",
                    message="模型返回的 tool_call 结构无效。",
                )
                self._append_tool_observation(call_id, "unknown", result, model_step, {})
                continue

            function = raw_call["function"]
            tool_name = function.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                result = ToolResult(
                    ok=False,
                    status="VALIDATION_ERROR",
                    error_code="MISSING_TOOL_NAME",
                    message="模型没有提供工具名称。",
                )
                self._append_tool_observation(call_id, "unknown", result, model_step, {})
                continue

            arguments, parse_error = self._parse_arguments(function.get("arguments"))
            if parse_error:
                result = ToolResult(
                    ok=False,
                    status="VALIDATION_ERROR",
                    error_code="INVALID_ARGUMENTS_JSON",
                    message=parse_error,
                )
            else:
                result = self.tool_executor.execute(tool_name, arguments)
            self._append_tool_observation(call_id, tool_name, result, model_step, arguments)

    def _append_tool_observation(
        self,
        call_id: str,
        tool_name: str,
        result: ToolResult,
        model_step: int,
        arguments: dict[str, Any],
    ) -> None:
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": tool_result_as_json(result),
            }
        )
        self._record(
            "tool",
            model_step,
            {
                "tool_call_id": call_id,
                "tool": tool_name,
                "arguments": arguments,
                "ok": result.ok,
                "status": result.status,
                "error_code": result.error_code,
            },
        )

    @staticmethod
    def _normalize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "role": "assistant",
            "content": message.get("content"),
        }
        if message.get("tool_calls") is not None:
            normalized["tool_calls"] = message["tool_calls"]
        if message.get("reasoning_content") is not None:
            normalized["reasoning_content"] = message["reasoning_content"]
        return normalized

    @staticmethod
    def _parse_arguments(raw_arguments: Any) -> tuple[dict[str, Any], str | None]:
        if isinstance(raw_arguments, dict):
            return raw_arguments, None
        if not isinstance(raw_arguments, str):
            return {}, "工具 arguments 必须是 JSON 字符串或对象。"
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {}, f"工具 arguments 不是合法 JSON：{exc.msg}。"
        if not isinstance(parsed, dict):
            return {}, "工具 arguments 必须解析为 JSON 对象。"
        return parsed, None

    @staticmethod
    def _tool_call_id(raw_call: Any, index: int) -> str:
        if isinstance(raw_call, dict) and isinstance(raw_call.get("id"), str):
            return raw_call["id"]
        return f"invalid_call_{index + 1}"

    def _record(self, event_type: str, model_step: int, detail: dict[str, Any]) -> None:
        self.trace.append(
            BaselineTraceEvent(
                sequence=len(self.trace) + 1,
                timestamp=datetime.now().isoformat(timespec="milliseconds"),
                event_type=event_type,
                model_step=model_step,
                detail=detail,
            )
        )
