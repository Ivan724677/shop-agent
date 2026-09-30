from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..agent_service import agent_service
from ..auth import CurrentUser, get_current_user, get_sse_user
from ..database import get_db
from ..db_models import AgentEventRecord, AgentRunRecord
from ..redis_bus import redis_bus
from ..schemas import (
    MessageCreate,
    MessageView,
    RunAccepted,
    RunView,
    SessionCreate,
    SessionView,
)


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionView, status_code=201)
async def create_session(
    body: SessionCreate,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await agent_service.create_session(db, user.user_id, body.title)


@router.get("", response_model=list[SessionView])
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await agent_service.list_sessions(db, user.user_id, limit)


@router.get("/{session_id}", response_model=SessionView)
async def get_session(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await agent_service.get_session(db, session_id, user.user_id)


@router.get("/{session_id}/messages", response_model=list[MessageView])
async def list_messages(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await agent_service.list_messages(db, session_id, user.user_id)


@router.post("/{session_id}/messages", response_model=RunAccepted, status_code=202)
async def send_message(
    session_id: str,
    body: MessageCreate,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await agent_service.enqueue_message(
        db,
        session_id,
        user.user_id,
        body.content,
        metadata={"client_message_id": body.client_message_id} if body.client_message_id else {},
    )
    return RunAccepted(run_id=run.id, session_id=run.session_id)


@router.get("/{session_id}/runs", response_model=list[RunView])
async def list_runs(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await agent_service.get_session(db, session_id, user.user_id)
    rows = await db.scalars(
        select(AgentRunRecord)
        .where(AgentRunRecord.session_id == session_id)
        .order_by(AgentRunRecord.created_at.desc())
        .limit(limit)
    )
    return list(rows)


@router.get("/{session_id}/runs/{run_id}", response_model=RunView)
async def get_run(
    session_id: str,
    run_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await agent_service.get_session(db, session_id, user.user_id)
    run = await db.get(AgentRunRecord, run_id)
    if run is None or run.session_id != session_id:
        from fastapi import HTTPException
        raise HTTPException(404, "Agent run 不存在。")
    return run


@router.get("/{session_id}/events")
async def stream_events(
    session_id: str,
    request: Request,
    after_sequence: int = Query(default=0, ge=0),
    user: CurrentUser = Depends(get_sse_user),
    db: AsyncSession = Depends(get_db),
):
    await agent_service.get_session(db, session_id, user.user_id)
    existing = list(await db.scalars(
        select(AgentEventRecord)
        .where(
            AgentEventRecord.session_id == session_id,
            AgentEventRecord.sequence > after_sequence,
        )
        .order_by(AgentEventRecord.sequence.asc())
        .limit(500)
    ))

    async def generate():
        for event in existing:
            payload = {
                "id": event.id,
                "sequence": event.sequence,
                "event": event.event_type,
                "payload": event.payload_json,
            }
            yield _sse(event.event_type, payload, event.sequence)
        async for event in redis_bus.subscribe(session_id):
            if await request.is_disconnected():
                break
            if event.get("event") == "heartbeat":
                yield ": heartbeat\n\n"
                continue
            yield _sse(
                str(event.get("event", "message")),
                event,
                int(event.get("sequence", 0)) or None,
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, payload: dict, sequence: int | None = None) -> str:
    lines = []
    if sequence is not None:
        lines.append(f"id: {sequence}")
    lines.append(f"event: {event}")
    lines.append("data: " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines) + "\n\n"

