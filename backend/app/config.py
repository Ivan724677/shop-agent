"""Environment-backed settings without importing secrets into application state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.environ.get("APP_NAME", "Risk-Aware Customer Service Agent")
    environment: str = os.environ.get("APP_ENV", "development")
    api_prefix: str = os.environ.get("API_PREFIX", "/api/v1")
    database_url: str = os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://agent:agent@postgres:5432/agent",
    )
    redis_url: str = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    cors_origins: tuple[str, ...] = tuple(
        value.strip()
        for value in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
        if value.strip()
    )
    dev_auth_enabled: bool = _bool("DEV_AUTH_ENABLED", True)
    offline_agent: bool = _bool("OFFLINE_AGENT", True)
    auto_create_schema: bool = _bool("AUTO_CREATE_SCHEMA", False)
    session_lock_seconds: int = int(os.environ.get("SESSION_LOCK_SECONDS", "60"))
    pending_action_ttl_seconds: int = int(os.environ.get("PENDING_ACTION_TTL_SECONDS", "300"))
    rag_index_root: Path = Path(os.environ.get("RAG_INDEX_ROOT", "var/rag_indexes"))
    metrics_namespace: str = os.environ.get("METRICS_NAMESPACE", "customer_service_agent")


settings = Settings()
