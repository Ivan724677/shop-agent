from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..agent_service import agent_service
from ..auth import CurrentUser, get_current_user
from ..database import get_db
from ..schemas import PendingActionCancel, PendingActionConfirm, PendingActionView, RunAccepted


router = APIRouter(tags=["pending-actions"])


@router.get("/sessions/{session_id}/pending-action", response_model=PendingActionView | None)
async def get_pending_action(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    found = await agent_service.get_current_pending_action(db, session_id, user.user_id)
    if found is None:
        return None
    record, token = found
    return PendingActionView.model_validate(record).model_copy(
        update={"confirmation_token": token}
    )


@router.post("/pending-actions/{action_id}/confirm", response_model=RunAccepted, status_code=202)
async def confirm_pending_action(
    action_id: str,
    body: PendingActionConfirm,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await agent_service.confirm_pending_action(
        db,
        action_id,
        user.user_id,
        body.confirmation_token,
        body.state_version,
    )
    return RunAccepted(run_id=run.id, session_id=run.session_id)


@router.post("/pending-actions/{action_id}/cancel", response_model=RunAccepted, status_code=202)
async def cancel_pending_action(
    action_id: str,
    body: PendingActionCancel,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    run = await agent_service.cancel_pending_action(
        db,
        action_id,
        user.user_id,
        body.state_version,
    )
    return RunAccepted(run_id=run.id, session_id=run.session_id)

