from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..agent_service import agent_service
from ..auth import CurrentUser, require_scope
from ..database import get_db
from ..db_models import ChatSessionRecord, HandoffRecord
from ..metrics import metrics
from ..schemas import HandoffClaim, HandoffResolve, HandoffView


router = APIRouter(prefix="/handoffs", tags=["handoffs"])


@router.get("", response_model=list[HandoffView])
async def list_handoffs(
    handoff_status: str = Query(default="open", alias="status"),
    user: CurrentUser = Depends(require_scope("handoff:manage")),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.scalars(
        select(HandoffRecord)
        .where(HandoffRecord.status == handoff_status)
        .order_by(HandoffRecord.created_at.asc())
        .limit(200)
    )
    return list(rows)


@router.post("/{handoff_id}/claim", response_model=HandoffView)
async def claim_handoff(
    handoff_id: str,
    body: HandoffClaim,
    user: CurrentUser = Depends(require_scope("handoff:manage")),
    db: AsyncSession = Depends(get_db),
):
    record = await db.get(HandoffRecord, handoff_id)
    if record is None:
        raise HTTPException(404, "人工接管记录不存在。")
    if record.status != "open":
        raise HTTPException(409, "该会话已被领取或处理。")
    record.status = "claimed"
    record.assigned_to = body.assignee or user.user_id
    await db.commit()
    await db.refresh(record)
    await agent_service.emit_event(
        db, record.session_id, record.run_id, "handoff.claimed",
        {"handoff_id": record.id, "assignee": record.assigned_to},
    )
    metrics.increment("handoffs_claimed_total")
    return record


@router.post("/{handoff_id}/resolve", response_model=HandoffView)
async def resolve_handoff(
    handoff_id: str,
    body: HandoffResolve,
    user: CurrentUser = Depends(require_scope("handoff:manage")),
    db: AsyncSession = Depends(get_db),
):
    record = await db.get(HandoffRecord, handoff_id)
    if record is None:
        raise HTTPException(404, "人工接管记录不存在。")
    if record.status not in {"open", "claimed"}:
        raise HTTPException(409, "该人工接管记录已结束。")
    record.status = "resolved"
    record.resolution = body.resolution
    record.resolved_at = datetime.now(timezone.utc)
    session = await db.get(ChatSessionRecord, record.session_id)
    if session is not None:
        session.status = "active"
    await db.commit()
    await db.refresh(record)
    await agent_service.emit_event(
        db, record.session_id, record.run_id, "handoff.resolved",
        {"handoff_id": record.id, "resolution": record.resolution},
    )
    metrics.increment("handoffs_resolved_total")
    return record

