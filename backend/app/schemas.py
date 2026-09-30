"""Pydantic HTTP contracts for the stage-seven API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SessionCreate(BaseModel):
    title: str = Field(default="售后咨询", max_length=200)


class SessionView(ORMModel):
    id: str
    user_id: str
    title: str
    status: str
    state_json: dict[str, Any]
    state_version: int
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=8000)
    client_message_id: str | None = Field(default=None, max_length=100)


class MessageView(ORMModel):
    id: str
    session_id: str
    role: str
    content: str
    sequence: int
    metadata_json: dict[str, Any]
    created_at: datetime


class RunAccepted(BaseModel):
    run_id: str
    session_id: str
    status: str = "queued"


class RunView(ORMModel):
    id: str
    session_id: str
    status: str
    current_node: str
    checkpoint_json: dict[str, Any]
    trace_json: list[dict[str, Any]]
    route_trace_json: list[dict[str, Any]]
    response_text: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class PendingActionView(ORMModel):
    id: str
    session_id: str
    action_type: str
    payload_json: dict[str, Any]
    state_version: int
    status: str
    expires_at: datetime
    confirmation_token: str | None = None


class PendingActionConfirm(BaseModel):
    confirmation_token: str = Field(min_length=20)
    state_version: int = Field(ge=0)


class PendingActionCancel(BaseModel):
    state_version: int = Field(ge=0)


class HandoffView(ORMModel):
    id: str
    session_id: str
    run_id: str | None
    ticket_id: str | None
    reason: str
    priority: str
    status: str
    assigned_to: str | None
    resolution: str | None
    created_at: datetime
    updated_at: datetime


class HandoffClaim(BaseModel):
    assignee: str = Field(min_length=1, max_length=64)


class HandoffResolve(BaseModel):
    resolution: str = Field(min_length=1, max_length=4000)


class FeedbackCreate(BaseModel):
    helpful: bool | None = None
    rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)
    expected_status: str | None = Field(default=None, max_length=32)
    expected_evidence_ids: list[str] = Field(default_factory=list, max_length=50)


class FeedbackView(ORMModel):
    id: str
    run_id: str
    session_id: str
    user_id: str
    helpful: bool | None
    rating: int | None
    comment: str | None
    created_at: datetime


class RAGIndexView(BaseModel):
    version: str
    status: Literal["built", "published", "archived", "missing"] | str
    current: bool
    manifest: dict[str, Any]


class HealthView(BaseModel):
    status: str
    database: str
    redis: str
    agent_mode: str

