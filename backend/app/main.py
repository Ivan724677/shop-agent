"""FastAPI entry point for the full-stack customer-service Agent."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import Base, engine
from . import db_models  # noqa: F401 - register ORM metadata for optional auto-create
from .redis_bus import redis_bus
from .routers import feedback, handoffs, operations, pending_actions, sessions


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_schema:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    try:
        await redis_bus.ensure_group()
    except Exception:
        # Readiness reports Redis as degraded; liveness remains available.
        pass
    yield
    await redis_bus.close()
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="7.0.0",
    lifespan=lifespan,
    openapi_url=f"{settings.api_prefix}/openapi.json",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions.router, prefix=settings.api_prefix)
app.include_router(pending_actions.router, prefix=settings.api_prefix)
app.include_router(handoffs.router, prefix=settings.api_prefix)
app.include_router(feedback.router, prefix=settings.api_prefix)
app.include_router(operations.router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "version": "7.0.0",
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health/ready",
    }
