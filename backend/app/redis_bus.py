"""Redis Streams jobs, Pub/Sub events and distributed session locks."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from .config import settings


JOB_STREAM = "agent:jobs"
JOB_GROUP = "agent-workers"


class RedisBus:
    def __init__(self, url: str | None = None) -> None:
        self.redis = Redis.from_url(url or settings.redis_url, decode_responses=True)

    async def ping(self) -> bool:
        return bool(await self.redis.ping())

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(JOB_STREAM, JOB_GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def enqueue_run(self, run_id: str) -> str:
        return str(await self.redis.xadd(JOB_STREAM, {"run_id": run_id}))

    async def read_jobs(
        self,
        consumer: str,
        *,
        block_ms: int = 5000,
        count: int = 10,
    ) -> list[tuple[str, str]]:
        await self.ensure_group()
        rows = await self.redis.xreadgroup(
            JOB_GROUP,
            consumer,
            {JOB_STREAM: ">"},
            count=count,
            block=block_ms,
        )
        output: list[tuple[str, str]] = []
        for _, messages in rows:
            for message_id, fields in messages:
                run_id = fields.get("run_id")
                if run_id:
                    output.append((message_id, str(run_id)))
        return output

    async def acknowledge_job(self, message_id: str) -> None:
        await self.redis.xack(JOB_STREAM, JOB_GROUP, message_id)

    async def publish(self, session_id: str, event: dict[str, Any]) -> None:
        await self.redis.publish(
            f"agent:events:{session_id}",
            json.dumps(event, ensure_ascii=False, separators=(",", ":")),
        )

    async def subscribe(self, session_id: str) -> AsyncIterator[dict[str, Any]]:
        pubsub = self.redis.pubsub()
        channel = f"agent:events:{session_id}"
        await pubsub.subscribe(channel)
        try:
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                if message and message.get("data"):
                    yield json.loads(str(message["data"]))
                else:
                    yield {"event": "heartbeat", "payload": {}}
                await asyncio.sleep(0)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    @asynccontextmanager
    async def session_lock(self, session_id: str) -> AsyncIterator[None]:
        lock = self.redis.lock(
            f"agent:lock:session:{session_id}",
            timeout=settings.session_lock_seconds,
            blocking_timeout=settings.session_lock_seconds,
        )
        acquired = await lock.acquire()
        if not acquired:
            raise TimeoutError("无法取得会话执行锁。")
        try:
            yield
        finally:
            try:
                await lock.release()
            except Exception:
                pass

    async def close(self) -> None:
        await self.redis.aclose()


redis_bus = RedisBus()


def worker_name() -> str:
    return f"worker-{uuid.uuid4().hex[:8]}"
