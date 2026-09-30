"""Minimal request identity boundary; replace with OIDC/JWT in deployment."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Query, status

from .config import settings


@dataclass(frozen=True)
class CurrentUser:
    user_id: str
    scopes: frozenset[str]


async def get_current_user(
    x_user_id: str | None = Header(default=None, alias="X-User-ID"),
    x_scopes: str | None = Header(default=None, alias="X-Scopes"),
) -> CurrentUser:
    if not x_user_id:
        if not settings.dev_auth_enabled:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "缺少认证身份。")
        x_user_id = "U001"
    scopes = frozenset(
        value.strip()
        for value in (x_scopes or "order:read,shipment:read,return:create,ticket:create,policy:read").split(",")
        if value.strip()
    )
    return CurrentUser(x_user_id, scopes)


async def get_sse_user(
    user_id: str | None = Query(default=None),
) -> CurrentUser:
    if not user_id and not settings.dev_auth_enabled:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "缺少 SSE 用户身份。")
    return CurrentUser(user_id or "U001", frozenset({"policy:read"}))


def require_scope(scope: str):
    async def dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if scope not in user.scopes:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"缺少权限：{scope}")
        return user

    return dependency

