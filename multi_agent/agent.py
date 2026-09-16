"""Stage-five agent assembly: state reduction, routing, experts, and MCP tools."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from baseline.tool_catalog import SessionToolExecutor
from experts import HandoffExpert, OrderLogisticsExpert, PolicyExpert, TransactionExpert
from experts.base import Expert
from routing.models import ExpertName, ExpertResult, ExpertStatus
from routing.router import MultiAgentRouter, RouteProtocolError
from rag.agent import AgenticRAG
from structured.extractor import DeterministicExtractor
from structured.models import SemanticFrame, StructuredTaskState, TaskStage, TurnMemory
from structured.reducer import StateReducer
from structured.semantic_parser import EmptySemanticParser, SemanticParser
from structured.state_machine import transition_state


@dataclass
class MultiAgentTraceEvent:
    sequence: int
    timestamp: str
    event_type: str
    turn: int
    detail: dict[str, Any] = field(default_factory=dict)


class MultiExpertCustomerServiceAgent:
    def __init__(
        self,
        semantic_parser: SemanticParser | None = None,
        tool_executor: SessionToolExecutor | None = None,
        router: MultiAgentRouter | None = None,
        experts: dict[ExpertName, Expert] | None = None,
        rag_agent: AgenticRAG | None = None,
        user_id: str = "U001",
        max_memory_turns: int = 50,
    ) -> None:
        if max_memory_turns <= 0:
            raise ValueError("max_memory_turns 必须大于 0。")
        self.tool_executor = tool_executor or SessionToolExecutor(user_id=user_id)
        self.semantic_parser = semantic_parser or EmptySemanticParser()
        self.extractor = DeterministicExtractor(self.tool_executor.store)
        self.reducer = StateReducer()
        self.router = router or MultiAgentRouter()
        self.state = StructuredTaskState(user_id=user_id)
        self.max_memory_turns = max_memory_turns
        self.memory: list[TurnMemory] = []
        self.trace: list[MultiAgentTraceEvent] = []
        self.route_trace: list[dict[str, Any]] = []
        self._turn_tool_names: list[str] = []
        self._active_expert: ExpertName | None = None
        self.experts: dict[ExpertName, Expert] = experts or {
            ExpertName.ORDER_LOGISTICS: OrderLogisticsExpert(self._call_tool),
            ExpertName.POLICY: PolicyExpert(self._call_tool, rag_agent=rag_agent),
            ExpertName.TRANSACTION: TransactionExpert(self._call_tool),
            ExpertName.HANDOFF: HandoffExpert(self._call_tool),
        }

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
        decision = self.router.decide(self.state, message, semantic)
        route_event = {
            **decision.as_dict(),
            "turn": self.state.turn_index,
            "input_state": self.state.snapshot(),
            "expert_results": [],
            "final_expert": None,
            "fallback_reason": None,
        }
        self.route_trace.append(route_event)
        self._record("route", decision.as_dict())
        response = self._dispatch(decision.experts, route_event)
        if not response:
            response = decision.clarification_message or (
                "当前请求无法安全路由，请补充订单号和具体诉求。"
            )
        self.state.last_response = response
        after = self.state.snapshot()
        self.memory.append(
            TurnMemory(
                turn=self.state.turn_index,
                user_message=message,
                semantic_frame=asdict(semantic),
                state_before=before,
                state_after=after,
                tool_names=list(self._turn_tool_names),
                response=response,
            )
        )
        if len(self.memory) > self.max_memory_turns:
            self.memory = self.memory[-self.max_memory_turns :]
        return response

    def _dispatch(
        self, candidates: list[ExpertName], route_event: dict[str, Any]
    ) -> str:
        if not candidates:
            return ""
        for index, expert_name in enumerate(candidates):
            remaining = candidates[index + 1 :]
            expert = self.experts.get(expert_name)
            if expert is None:
                return self._protocol_fallback(
                    route_event, f"缺少专家实现：{expert_name.value}"
                )
            self._active_expert = expert_name
            try:
                result = expert.execute(self.state)
                result = self.router.validate_expert_result(
                    result, expert_name, remaining
                )
            except Exception as exc:
                self._active_expert = None
                return self._protocol_fallback(route_event, str(exc))
            finally:
                if self._active_expert == expert_name:
                    self._active_expert = None
            self._record_expert_result(route_event, result)
            if result.status == ExpertStatus.CONTINUE:
                continue
            if result.status == ExpertStatus.ESCALATE:
                return self._handoff(route_event, result.escalation_reason)
            route_event["final_expert"] = expert_name.value
            return result.response
        return self._protocol_fallback(route_event, "专家链结束但没有终止结果")

    def _handoff(self, route_event: dict[str, Any], reason: str) -> str:
        expert = self.experts.get(ExpertName.HANDOFF)
        if not isinstance(expert, HandoffExpert):
            return self._protocol_fallback(
                route_event, "人工专家不可用", allow_handoff=False
            )
        self._active_expert = ExpertName.HANDOFF
        try:
            result = expert.execute(self.state, reason=reason)
            result = self.router.validate_expert_result(
                result, ExpertName.HANDOFF, []
            )
        except Exception as exc:
            return self._protocol_fallback(
                route_event, str(exc), allow_handoff=False
            )
        finally:
            self._active_expert = None
        self._record_expert_result(route_event, result)
        route_event["final_expert"] = ExpertName.HANDOFF.value
        if route_event["fallback_reason"] is None:
            route_event["fallback_reason"] = reason
        return result.response

    def _protocol_fallback(
        self,
        route_event: dict[str, Any],
        reason: str,
        allow_handoff: bool = True,
    ) -> str:
        route_event["fallback_reason"] = f"EXPERT_PROTOCOL_ERROR: {reason}"
        self._record("route_protocol_error", {"reason": reason})
        if allow_handoff:
            return self._handoff(route_event, f"专家输出协议错误：{reason}")
        transition_state(self.state, TaskStage.HANDOFF)
        return "自动处理链路异常，且暂时无法创建人工工单。请稍后联系人工客服。"

    def _record_expert_result(
        self, route_event: dict[str, Any], result: ExpertResult
    ) -> None:
        detail = {
            "expert": result.expert.value,
            "status": result.status.value,
            "next_expert": result.next_expert.value if result.next_expert else None,
            "reason_code": result.reason_code,
            "escalation_reason": result.escalation_reason,
            "facts": result.facts,
        }
        route_event["expert_results"].append(detail)
        self._record("expert", detail)

    def _call_tool(self, tool_name: str, arguments: dict) -> Any:
        result = self.tool_executor.execute(tool_name, arguments)
        self._turn_tool_names.append(tool_name)
        self._record(
            "tool",
            {
                "expert": self._active_expert.value if self._active_expert else None,
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

    def reset(self) -> None:
        user_id = self.state.user_id
        self.state = StructuredTaskState(user_id=user_id)
        self.memory = []
        self.trace = []
        self.route_trace = []

    def state_as_json(self) -> str:
        return json.dumps(self.state.snapshot(), ensure_ascii=False, indent=2)

    def memory_as_json(self) -> str:
        return json.dumps([asdict(turn) for turn in self.memory], ensure_ascii=False, indent=2)

    def trace_as_json(self) -> str:
        return json.dumps([asdict(event) for event in self.trace], ensure_ascii=False, indent=2)

    def route_trace_as_json(self) -> str:
        return json.dumps(self.route_trace, ensure_ascii=False, indent=2)

    def _record(self, event_type: str, detail: dict[str, Any]) -> None:
        self.trace.append(
            MultiAgentTraceEvent(
                sequence=len(self.trace) + 1,
                timestamp=datetime.now().isoformat(timespec="milliseconds"),
                event_type=event_type,
                turn=self.state.turn_index,
                detail=detail,
            )
        )
