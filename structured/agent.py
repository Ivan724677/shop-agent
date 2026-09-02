"""阶段三结构化 Agent 装配层。

把感知（extractor + semantic_parser）、状态决策（reducer）、
行动决策（policy）串成一个完整的多轮对话 Agent：
handle() 每轮执行 感知→归并→策略→记录 trace 与情节记忆。

复用阶段二的 SessionToolExecutor 绑定会话身份；
情节记忆（TurnMemory）限量 max_memory_turns，超限自动裁剪。
对比阶段二 baseline：LLM 只做语义解析，工具决策与状态管理由确定性代码接管。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from baseline.tool_catalog import SessionToolExecutor

from .extractor import DeterministicExtractor
from .models import SemanticFrame, StructuredTaskState, TurnMemory
from .policy import DialoguePolicy
from .reducer import StateReducer
from .semantic_parser import EmptySemanticParser, SemanticParser


@dataclass
class StructuredTraceEvent:
    sequence: int
    timestamp: str
    event_type: str
    turn: int
    detail: dict[str, Any] = field(default_factory=dict)


class StructuredCustomerServiceAgent:
    def __init__(
        self,
        semantic_parser: SemanticParser | None = None,
        tool_executor: SessionToolExecutor | None = None,
        user_id: str = "U001",
        max_memory_turns: int = 50,
    ) -> None:
        if max_memory_turns <= 0:
            raise ValueError("max_memory_turns 必须大于 0。")
        self.tool_executor = tool_executor or SessionToolExecutor(user_id=user_id)
        self.semantic_parser = semantic_parser or EmptySemanticParser()
        self.extractor = DeterministicExtractor(self.tool_executor.store)
        self.reducer = StateReducer()
        self.state = StructuredTaskState(user_id=user_id)
        self.max_memory_turns = max_memory_turns
        self.memory: list[TurnMemory] = []
        self.trace: list[StructuredTraceEvent] = []
        self._turn_tool_names: list[str] = []
        self.policy = DialoguePolicy(self._call_tool)

    def handle(self, user_message: str) -> str:
        if not user_message.strip():
            raise ValueError("user_message 不能为空。")
        message = user_message.strip()
        before = self.state.snapshot()
        deterministic = self.extractor.extract(message)
        parser_error: str | None = None
        try:
            semantic = self.semantic_parser.parse(message, self.state.parser_context())
        except (TypeError, ValueError) as exc:
            semantic = SemanticFrame()
            parser_error = str(exc)

        self.reducer.reduce(self.state, message, deterministic, semantic)
        if parser_error:
            self.state.conflicts.append("SEMANTIC_PARSER_ERROR")
        self._record(
            "parse",
            {
                "deterministic": asdict(deterministic),
                "semantic": asdict(semantic),
                "parser_error": parser_error,
                "usage": getattr(self.semantic_parser, "last_usage", {}),
            },
        )
        self._record("state_reduced", self.state.snapshot())

        self._turn_tool_names = []
        outcome = self.policy.decide(self.state)
        self.state.last_response = outcome.response
        after = self.state.snapshot()
        self._record(
            "policy",
            {
                "stage": outcome.final_state.value,
                "response": outcome.response,
                "missing_fields": self.state.missing_fields,
            },
        )
        self.memory.append(
            TurnMemory(
                turn=self.state.turn_index,
                user_message=message,
                semantic_frame=asdict(semantic),
                state_before=before,
                state_after=after,
                tool_names=list(self._turn_tool_names),
                response=outcome.response,
            )
        )
        if len(self.memory) > self.max_memory_turns:
            self.memory = self.memory[-self.max_memory_turns :]
        return outcome.response

    def reset(self) -> None:
        user_id = self.state.user_id
        self.state = StructuredTaskState(user_id=user_id)
        self.memory = []
        self.trace = []

    def state_as_json(self) -> str:
        return json.dumps(self.state.snapshot(), ensure_ascii=False, indent=2)

    def memory_as_json(self) -> str:
        return json.dumps([asdict(turn) for turn in self.memory], ensure_ascii=False, indent=2)

    def trace_as_json(self) -> str:
        return json.dumps([asdict(event) for event in self.trace], ensure_ascii=False, indent=2)

    def _call_tool(self, tool_name: str, arguments: dict) -> Any:
        result = self.tool_executor.execute(tool_name, arguments)
        self._turn_tool_names.append(tool_name)
        self._record(
            "tool",
            {
                "tool": tool_name,
                "arguments": arguments,
                "ok": result.ok,
                "status": result.status,
                "error_code": result.error_code,
                "audit_id": result.metadata.get("audit_id"),
                "attempts": result.metadata.get("attempts", 0),
                "reconciliation_tool": result.metadata.get("reconciliation_tool"),
            },
        )
        return result

    def _record(self, event_type: str, detail: dict[str, Any]) -> None:
        self.trace.append(
            StructuredTraceEvent(
                sequence=len(self.trace) + 1,
                timestamp=datetime.now().isoformat(timespec="milliseconds"),
                event_type=event_type,
                turn=self.state.turn_index,
                detail=detail,
            )
        )
