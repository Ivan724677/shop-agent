"""Persistent session API and asynchronous Agent run execution."""

from __future__ import annotations

import asyncio
import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from structured.models import TaskStage

from .agent_runtime import runtime_registry
from .config import settings
from .db_models import (
    AgentEventRecord,
    AgentRunRecord,
    ChatMessageRecord,
    ChatSessionRecord,
    FeedbackRecord,
    HandoffRecord,
    PendingActionRecord,
)
from .metrics import metrics
from .redis_bus import RedisBus, redis_bus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def token_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def action_scope_hash(payload: dict[str, Any]) -> str:
    canonical = "|".join([
        str(payload.get("action", "")),
        str(payload.get("order_id", "")),
        ",".join(sorted(str(value) for value in payload.get("item_ids", []))),
        f"{float(payload.get('amount', 0.0)):.2f}",
        str(payload.get("reason", "")),
    ])
    return token_hash(canonical)


class AgentApplicationService:
    def __init__(self, bus: RedisBus | None = None) -> None:
        self.bus = bus or redis_bus

    async def create_session(
        self,
        db: AsyncSession,
        user_id: str,
        title: str,
    ) -> ChatSessionRecord:
        record = ChatSessionRecord(user_id=user_id, title=title)
        db.add(record)
        await db.commit()
        await db.refresh(record)
        metrics.increment("sessions_created_total")
        return record

    async def get_session(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        *,
        allow_staff: bool = False,
    ) -> ChatSessionRecord:
        record = await db.get(ChatSessionRecord, session_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "会话不存在。")
        if not allow_staff and record.user_id != user_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "无权访问该会话。")
        return record

    async def list_sessions(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 50,
    ) -> list[ChatSessionRecord]:
        rows = await db.scalars(
            select(ChatSessionRecord)
            .where(ChatSessionRecord.user_id == user_id)
            .order_by(ChatSessionRecord.updated_at.desc())
            .limit(limit)
        )
        return list(rows)

    async def list_messages(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str,
    ) -> list[ChatMessageRecord]:
        await self.get_session(db, session_id, user_id)
        rows = await db.scalars(
            select(ChatMessageRecord)
            .where(ChatMessageRecord.session_id == session_id)
            .order_by(ChatMessageRecord.sequence.asc())
        )
        return list(rows)

    async def enqueue_message(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRunRecord:
        session = await self.get_session(db, session_id, user_id)
        if session.status not in {"active", "waiting_for_user"}:
            raise HTTPException(status.HTTP_409_CONFLICT, "当前会话不可接收新消息。")
        max_sequence = await db.scalar(
            select(func.coalesce(func.max(ChatMessageRecord.sequence), 0)).where(
                ChatMessageRecord.session_id == session_id
            )
        )
        message = ChatMessageRecord(
            session_id=session_id,
            role="user",
            content=content.strip(),
            sequence=int(max_sequence or 0) + 1,
            metadata_json=metadata or {},
        )
        db.add(message)
        await db.flush()
        run = AgentRunRecord(
            session_id=session_id,
            input_message_id=message.id,
            status="queued",
            current_node="queued",
            checkpoint_json=session.state_json or {},
        )
        db.add(run)
        await db.commit()
        await self.bus.enqueue_run(run.id)
        await self.emit_event(
            db,
            session_id,
            run.id,
            "run.queued",
            {"run_id": run.id, "message_id": message.id},
        )
        metrics.increment("messages_received_total")
        metrics.increment("agent_runs_queued_total")
        return run

    async def process_run(self, db: AsyncSession, run_id: str) -> None:
        run = await db.get(AgentRunRecord, run_id)
        if run is None or run.status not in {"queued", "retry"}:
            return
        session = await db.get(ChatSessionRecord, run.session_id)
        message = await db.get(ChatMessageRecord, run.input_message_id) if run.input_message_id else None
        if session is None or message is None:
            run.status = "failed"
            run.error_message = "会话或输入消息不存在。"
            await db.commit()
            return

        async with self.bus.session_lock(session.id):
            run.status = "running"
            run.current_node = "agent.handle"
            run.started_at = utcnow()
            await db.commit()
            await self.emit_event(db, session.id, run.id, "run.started", {"run_id": run.id})
            try:
                agent = runtime_registry.get_or_create(
                    session.id,
                    session.user_id,
                    session.state_json or run.checkpoint_json,
                )
                trace_start = len(agent.trace)
                route_start = len(agent.route_trace)
                response = await asyncio.to_thread(agent.handle, message.content)
                checkpoint = runtime_registry.checkpoint(agent)

                max_sequence = await db.scalar(
                    select(func.coalesce(func.max(ChatMessageRecord.sequence), 0)).where(
                        ChatMessageRecord.session_id == session.id
                    )
                )
                assistant_message = ChatMessageRecord(
                    session_id=session.id,
                    role="assistant",
                    content=response,
                    sequence=int(max_sequence or 0) + 1,
                    metadata_json={
                        "run_id": run.id,
                        "policy_evidence": list(agent.state.policy_evidence),
                        "stage": agent.state.stage.value,
                    },
                )
                db.add(assistant_message)
                session.state_json = checkpoint
                session.state_version += 1
                session.status = (
                    "waiting_for_human"
                    if agent.state.stage == TaskStage.HANDOFF
                    else "waiting_for_user"
                    if agent.state.pending_action is not None
                    else "active"
                )
                run.status = "completed"
                run.current_node = "completed"
                run.checkpoint_json = checkpoint
                run.trace_json = [
                    _jsonable({
                        "sequence": event.sequence,
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "turn": event.turn,
                        "detail": event.detail,
                    })
                    for event in agent.trace[trace_start:]
                ]
                run.route_trace_json = [_jsonable(dict(row)) for row in agent.route_trace[route_start:]]
                run.response_text = response
                run.completed_at = utcnow()
                pending_payload = None
                confirmation_token = None
                if agent.state.pending_action is not None:
                    pending_payload = asdict_pending(agent.state.pending_action)
                    confirmation_token = await self._create_pending_action(
                        db,
                        session,
                        run,
                        pending_payload,
                    )
                if agent.state.stage == TaskStage.HANDOFF:
                    await self._create_handoff(db, session, run, response)
                await self._settle_previous_action(db, session.id, message.metadata_json)
                await db.commit()

                for event in run.trace_json:
                    await self.emit_event(
                        db,
                        session.id,
                        run.id,
                        f"agent.{event['event_type']}",
                        event,
                    )
                await self.emit_event(
                    db,
                    session.id,
                    run.id,
                    "message.completed",
                    {
                        "run_id": run.id,
                        "message": {
                            "id": assistant_message.id,
                            "role": "assistant",
                            "content": response,
                            "sequence": assistant_message.sequence,
                            "metadata": assistant_message.metadata_json,
                        },
                        "pending_action": (
                            {
                                "id": pending_payload["record_id"],
                                "action_type": pending_payload["action"],
                                "payload": pending_payload,
                                "state_version": session.state_version,
                                "confirmation_token": confirmation_token,
                            }
                            if pending_payload else None
                        ),
                        "handoff": agent.state.stage == TaskStage.HANDOFF,
                    },
                )
                metrics.increment("agent_runs_completed_total")
                if agent.state.pending_action is not None:
                    metrics.increment("pending_actions_created_total")
                if agent.state.stage == TaskStage.HANDOFF:
                    metrics.increment("handoffs_created_total")
            except Exception as exc:
                run.status = "failed"
                run.current_node = "failed"
                run.error_message = f"{type(exc).__name__}: {exc}"
                run.completed_at = utcnow()
                session.status = "active"
                await db.commit()
                await self.emit_event(
                    db,
                    session.id,
                    run.id,
                    "run.failed",
                    {"run_id": run.id, "error": "Agent 处理失败，请稍后重试或联系人工。"},
                )
                metrics.increment("agent_runs_failed_total")

    async def emit_event(
        self,
        db: AsyncSession,
        session_id: str,
        run_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> AgentEventRecord:
        maximum = await db.scalar(
            select(func.coalesce(func.max(AgentEventRecord.sequence), 0)).where(
                AgentEventRecord.session_id == session_id
            )
        )
        event = AgentEventRecord(
            session_id=session_id,
            run_id=run_id,
            sequence=int(maximum or 0) + 1,
            event_type=event_type,
            payload_json=payload,
        )
        db.add(event)
        await db.commit()
        await self.bus.publish(
            session_id,
            {
                "id": event.id,
                "sequence": event.sequence,
                "event": event.event_type,
                "payload": event.payload_json,
                "created_at": event.created_at.isoformat() if event.created_at else utcnow().isoformat(),
            },
        )
        return event

    async def get_current_pending_action(
        self,
        db: AsyncSession,
        session_id: str,
        user_id: str,
    ) -> tuple[PendingActionRecord, str] | None:
        await self.get_session(db, session_id, user_id)
        record = await db.scalar(
            select(PendingActionRecord)
            .where(
                PendingActionRecord.session_id == session_id,
                PendingActionRecord.status == "pending",
            )
            .order_by(PendingActionRecord.created_at.desc())
            .limit(1)
        )
        if record is None or record.expires_at <= utcnow():
            if record is not None:
                record.status = "expired"
                await db.commit()
            return None
        token = secrets.token_urlsafe(32)
        record.confirmation_token_hash = token_hash(token)
        await db.commit()
        return record, token

    async def confirm_pending_action(
        self,
        db: AsyncSession,
        action_id: str,
        user_id: str,
        confirmation_token: str,
        state_version: int,
    ) -> AgentRunRecord:
        action = await db.get(PendingActionRecord, action_id)
        if action is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "待确认操作不存在。")
        session = await self.get_session(db, action.session_id, user_id)
        if action.status != "pending" or action.expires_at <= utcnow():
            raise HTTPException(status.HTTP_409_CONFLICT, "待确认操作已失效或已处理。")
        if session.state_version != state_version or action.state_version != state_version:
            raise HTTPException(status.HTTP_409_CONFLICT, "会话状态已变化，请重新确认最新操作。")
        if not secrets.compare_digest(action.confirmation_token_hash, token_hash(confirmation_token)):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "确认令牌无效。")
        action.status = "confirmation_received"
        action.confirmed_at = utcnow()
        await db.commit()
        return await self.enqueue_message(
            db,
            session.id,
            user_id,
            "确认退货",
            metadata={"source": "pending_action_confirmation", "pending_action_id": action.id},
        )

    async def cancel_pending_action(
        self,
        db: AsyncSession,
        action_id: str,
        user_id: str,
        state_version: int,
    ) -> AgentRunRecord:
        action = await db.get(PendingActionRecord, action_id)
        if action is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "待确认操作不存在。")
        session = await self.get_session(db, action.session_id, user_id)
        if action.status != "pending" or session.state_version != state_version:
            raise HTTPException(status.HTTP_409_CONFLICT, "待确认操作已失效或会话状态已变化。")
        action.status = "cancellation_received"
        await db.commit()
        return await self.enqueue_message(
            db,
            session.id,
            user_id,
            "取消退货",
            metadata={"source": "pending_action_cancel", "pending_action_id": action.id},
        )

    async def _create_pending_action(
        self,
        db: AsyncSession,
        session: ChatSessionRecord,
        run: AgentRunRecord,
        payload: dict[str, Any],
    ) -> str:
        await db.execute(
            update(PendingActionRecord)
            .where(
                PendingActionRecord.session_id == session.id,
                PendingActionRecord.status == "pending",
            )
            .values(status="superseded")
        )
        token = secrets.token_urlsafe(32)
        record = PendingActionRecord(
            session_id=session.id,
            run_id=run.id,
            action_type=str(payload["action"]),
            payload_json=payload,
            scope_hash=action_scope_hash(payload),
            confirmation_token_hash=token_hash(token),
            state_version=session.state_version,
            expires_at=utcnow() + timedelta(seconds=settings.pending_action_ttl_seconds),
        )
        db.add(record)
        await db.flush()
        payload["record_id"] = record.id
        return token

    async def _settle_previous_action(
        self,
        db: AsyncSession,
        session_id: str,
        metadata: dict[str, Any],
    ) -> None:
        action_id = metadata.get("pending_action_id") if isinstance(metadata, dict) else None
        if not action_id:
            return
        action = await db.get(PendingActionRecord, str(action_id))
        if action is not None and action.session_id == session_id:
            action.status = (
                "cancelled" if metadata.get("source") == "pending_action_cancel" else "completed"
            )

    async def _create_handoff(
        self,
        db: AsyncSession,
        session: ChatSessionRecord,
        run: AgentRunRecord,
        response: str,
    ) -> None:
        existing = await db.scalar(
            select(HandoffRecord).where(
                HandoffRecord.session_id == session.id,
                HandoffRecord.status.in_(["open", "claimed"]),
            )
        )
        if existing is not None:
            return
        match = re.search(r"\bT\d{5}\b", response)
        reason = "Agent 无法安全完成自动处理"
        if run.route_trace_json:
            reason = str(run.route_trace_json[-1].get("fallback_reason") or reason)
        db.add(HandoffRecord(
            session_id=session.id,
            run_id=run.id,
            ticket_id=match.group(0) if match else None,
            reason=reason,
            priority="high" if "未知" in reason or "协议" in reason else "normal",
        ))


def asdict_pending(pending: Any) -> dict[str, Any]:
    return {
        "action": pending.action,
        "order_id": pending.order_id,
        "item_ids": list(pending.item_ids),
        "amount": float(pending.amount),
        "reason": pending.reason,
        "idempotency_key": pending.idempotency_key,
        "requested_turn": pending.requested_turn,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value


agent_service = AgentApplicationService()
