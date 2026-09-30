from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rag.index import IndexLifecycleError, PersistentIndex

from ..auth import CurrentUser, require_scope
from ..config import settings
from ..database import get_db
from ..db_models import RAGIndexRecord
from ..metrics import metrics
from ..redis_bus import redis_bus
from ..schemas import HealthView, RAGIndexView


router = APIRouter(tags=["operations"])


@router.get("/health/live", response_model=HealthView)
async def live(db: AsyncSession = Depends(get_db)):
    return HealthView(
        status="live",
        database="unknown",
        redis="unknown",
        agent_mode="offline" if settings.offline_agent else "online",
    )


@router.get("/health/ready", response_model=HealthView)
async def ready(db: AsyncSession = Depends(get_db)):
    database = "ready"
    redis = "ready"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        database = "unavailable"
    try:
        await redis_bus.ping()
    except Exception:
        redis = "unavailable"
    overall = "ready" if database == redis == "ready" else "degraded"
    return HealthView(
        status=overall,
        database=database,
        redis=redis,
        agent_mode="offline" if settings.offline_agent else "online",
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@router.get("/rag/indexes", response_model=list[RAGIndexView])
async def list_indexes(
    user: CurrentUser = Depends(require_scope("rag:manage")),
    db: AsyncSession = Depends(get_db),
):
    index = PersistentIndex(settings.rag_index_root)
    current = index.current_version()
    versions = index.list_versions()
    output: list[RAGIndexView] = []
    for version in versions:
        try:
            manifest, _ = index.load(version)
            manifest_json = manifest.as_dict()
        except IndexLifecycleError:
            manifest_json = {}
        status_value = "published" if version == current else "built"
        record = await db.scalar(select(RAGIndexRecord).where(RAGIndexRecord.version == version))
        if record is None:
            record = RAGIndexRecord(
                version=version,
                status=status_value,
                manifest_json=manifest_json,
                published_at=datetime.now(timezone.utc) if version == current else None,
            )
            db.add(record)
        else:
            record.status = status_value
            record.manifest_json = manifest_json
        output.append(RAGIndexView(
            version=version,
            status=status_value,
            current=version == current,
            manifest=manifest_json,
        ))
    await db.commit()
    return output


@router.post("/rag/indexes/{version}/publish", response_model=RAGIndexView)
async def publish_index(
    version: str,
    user: CurrentUser = Depends(require_scope("rag:manage")),
    db: AsyncSession = Depends(get_db),
):
    index = PersistentIndex(settings.rag_index_root)
    try:
        index.publish(version)
        manifest, _ = index.load(version)
    except IndexLifecycleError as exc:
        raise HTTPException(404, str(exc)) from exc
    rows = list(await db.scalars(select(RAGIndexRecord)))
    for row in rows:
        row.status = "published" if row.version == version else "built"
    record = next((row for row in rows if row.version == version), None)
    if record is None:
        record = RAGIndexRecord(version=version)
        db.add(record)
    record.status = "published"
    record.manifest_json = manifest.as_dict()
    record.published_at = datetime.now(timezone.utc)
    await db.commit()
    metrics.increment("rag_index_publish_total")
    return RAGIndexView(version=version, status="published", current=True, manifest=manifest.as_dict())


@router.post("/rag/indexes/{version}/rollback", response_model=RAGIndexView)
async def rollback_index(
    version: str,
    user: CurrentUser = Depends(require_scope("rag:manage")),
    db: AsyncSession = Depends(get_db),
):
    result = await publish_index(version, user, db)
    metrics.increment("rag_index_rollback_total")
    return result

