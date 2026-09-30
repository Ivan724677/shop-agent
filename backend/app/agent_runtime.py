"""Adapter between persistent web sessions and the existing stateful agent core."""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from agent import PolicyKnowledge, ToolGateway
from baseline.deepseek_client import DeepSeekClient
from baseline.tool_catalog import SessionToolExecutor
from mcp_server.server import MCPToolServer
from multi_agent.agent import MultiAgentTraceEvent, MultiExpertCustomerServiceAgent
from rag.agent import AgenticRAG
from rag.generation import DeepSeekAnswerGenerator
from rag.grading import DeepSeekDocumentGrader
from rag.planner import DeepSeekRAGPlanner
from rag.retrieval import load_policy_corpus
from repositories import InMemoryStore
from routing.router import MultiAgentRouter
from routing.semantic_router import DeepSeekSemanticRouter
from structured.models import (
    ConfirmationStatus,
    Intent,
    PendingAction,
    RiskLevel,
    SlotSource,
    SlotValue,
    StructuredTaskState,
    TaskStage,
    ToolFact,
    TurnMemory,
)
from structured.semantic_parser import DeepSeekSemanticParser, EmptySemanticParser

from .config import settings


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


class AgentRuntimeRegistry:
    """One in-process runtime per active session, checkpointed after every turn.

    PostgreSQL remains the source of truth for conversation/run state. Existing
    stage-one business tools still use their tested repository contract; the
    registry shares one tool backend within a worker so authenticated sessions
    observe consistent seed and write state.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: dict[str, MultiExpertCustomerServiceAgent] = {}
        self.store = InMemoryStore()
        self.gateway = ToolGateway(self.store, PolicyKnowledge())
        self.mcp_server = MCPToolServer(self.gateway)

    def get_or_create(
        self,
        session_id: str,
        user_id: str,
        checkpoint: dict[str, Any] | None = None,
    ) -> MultiExpertCustomerServiceAgent:
        with self._lock:
            agent = self._agents.get(session_id)
            if agent is not None:
                return agent
            executor = SessionToolExecutor(
                user_id=user_id,
                store=self.store,
                gateway=self.gateway,
                mcp_server=self.mcp_server,
                session_id=session_id,
            )
            if settings.offline_agent or not os.environ.get("DEEPSEEK_API_KEY"):
                agent = MultiExpertCustomerServiceAgent(
                    semantic_parser=EmptySemanticParser(),
                    tool_executor=executor,
                    router=MultiAgentRouter(),
                    user_id=user_id,
                )
            else:
                client = DeepSeekClient.from_env()
                rag = AgenticRAG(
                    load_policy_corpus(),
                    planner=DeepSeekRAGPlanner(client),
                    grader=DeepSeekDocumentGrader(client),
                    generator=DeepSeekAnswerGenerator(client),
                )
                agent = MultiExpertCustomerServiceAgent(
                    semantic_parser=DeepSeekSemanticParser(client),
                    tool_executor=executor,
                    router=MultiAgentRouter(DeepSeekSemanticRouter(client)),
                    rag_agent=rag,
                    user_id=user_id,
                )
            agent.state.session_id = session_id
            if checkpoint:
                self.restore(agent, checkpoint)
            self._agents[session_id] = agent
            return agent

    def checkpoint(self, agent: MultiExpertCustomerServiceAgent) -> dict[str, Any]:
        return {
            "state": _jsonable(agent.state),
            "memory": _jsonable(agent.memory),
            "trace": _jsonable(agent.trace),
            "route_trace": _jsonable(agent.route_trace),
        }

    def restore(self, agent: MultiExpertCustomerServiceAgent, checkpoint: dict[str, Any]) -> None:
        raw = checkpoint.get("state")
        if not isinstance(raw, dict):
            return
        state = StructuredTaskState(
            user_id=str(raw.get("user_id", agent.state.user_id)),
            session_id=str(raw.get("session_id", agent.state.session_id)),
            task_id=str(raw.get("task_id", agent.state.task_id)),
            turn_index=int(raw.get("turn_index", 0)),
            intent=Intent(str(raw.get("intent", Intent.UNKNOWN.value))),
            stage=TaskStage(str(raw.get("stage", TaskStage.IDLE.value))),
            risk_level=RiskLevel(str(raw.get("risk_level", RiskLevel.LOW.value))),
            selected_item_ids=[str(value) for value in raw.get("selected_item_ids", [])],
            excluded_item_ids=[str(value) for value in raw.get("excluded_item_ids", [])],
            missing_fields=[str(value) for value in raw.get("missing_fields", [])],
            confirmation_status=ConfirmationStatus(
                str(raw.get("confirmation_status", ConfirmationStatus.NOT_REQUESTED.value))
            ),
            policy_evidence=[str(value) for value in raw.get("policy_evidence", [])],
            conflicts=[str(value) for value in raw.get("conflicts", [])],
            consultation_only=bool(raw.get("consultation_only", False)),
            last_response=str(raw.get("last_response", "")),
        )
        for name, slot in dict(raw.get("slots", {})).items():
            if not isinstance(slot, dict):
                continue
            state.slots[str(name)] = SlotValue(
                value=slot.get("value"),
                source=SlotSource(str(slot.get("source", SlotSource.DERIVED.value))),
                confidence=float(slot.get("confidence", 0.0)),
                evidence=str(slot.get("evidence", "checkpoint")),
                updated_turn=int(slot.get("updated_turn", 0)),
            )
        pending = raw.get("pending_action")
        if isinstance(pending, dict):
            state.pending_action = PendingAction(
                action=str(pending["action"]),
                order_id=str(pending["order_id"]),
                item_ids=[str(value) for value in pending.get("item_ids", [])],
                amount=float(pending["amount"]),
                reason=str(pending["reason"]),
                idempotency_key=str(pending["idempotency_key"]),
                requested_turn=int(pending["requested_turn"]),
            )
        for key, fact in dict(raw.get("facts", {})).items():
            if isinstance(fact, dict):
                state.facts[str(key)] = ToolFact(
                    key=str(fact.get("key", key)),
                    value=fact.get("value"),
                    tool=str(fact.get("tool", "checkpoint")),
                    observed_turn=int(fact.get("observed_turn", 0)),
                )
        agent.state = state
        agent.memory = [TurnMemory(**row) for row in checkpoint.get("memory", []) if isinstance(row, dict)]
        agent.trace = [
            MultiAgentTraceEvent(**row)
            for row in checkpoint.get("trace", [])
            if isinstance(row, dict)
        ]
        agent.route_trace = [
            dict(row) for row in checkpoint.get("route_trace", []) if isinstance(row, dict)
        ]

    def discard(self, session_id: str) -> None:
        with self._lock:
            self._agents.pop(session_id, None)


runtime_registry = AgentRuntimeRegistry()

