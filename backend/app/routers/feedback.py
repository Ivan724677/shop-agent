from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import CurrentUser, get_current_user
from ..database import get_db
from ..db_models import AgentRunRecord, ChatSessionRecord, FeedbackRecord
from ..metrics import metrics
from ..schemas import FeedbackCreate, FeedbackView


router = APIRouter(prefix="/runs", tags=["feedback"])


@router.post("/{run_id}/feedback", response_model=FeedbackView, status_code=201)
async def create_feedback(
    run_id: str,
    body: FeedbackCreate,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await db.get(AgentRunRecord, run_id)
    if run is None:
        raise HTTPException(404, "Agent run 不存在。")
    session = await db.get(ChatSessionRecord, run.session_id)
    if session is None or session.user_id != user.user_id:
        raise HTTPException(403, "无权评价该 Agent run。")
    existing = await db.scalar(
        select(FeedbackRecord).where(
            FeedbackRecord.run_id == run_id,
            FeedbackRecord.user_id == user.user_id,
        )
    )
    if existing is not None:
        raise HTTPException(409, "该运行已经提交过反馈。")
    record = FeedbackRecord(
        run_id=run_id,
        session_id=run.session_id,
        user_id=user.user_id,
        helpful=body.helpful,
        rating=body.rating,
        comment=body.comment,
        expected_status=body.expected_status,
        expected_evidence_ids=body.expected_evidence_ids,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    metrics.increment("feedback_total")
    if body.helpful is True:
        metrics.increment("feedback_helpful_total")
    return record

