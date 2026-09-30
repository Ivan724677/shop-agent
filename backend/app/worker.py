"""Redis Stream worker for durable Agent runs."""

from __future__ import annotations

import asyncio
import logging

from .agent_service import agent_service
from .database import AsyncSessionFactory
from .redis_bus import redis_bus, worker_name


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-worker")


async def run_worker() -> None:
    consumer = worker_name()
    await redis_bus.ensure_group()
    logger.info("agent worker started consumer=%s", consumer)
    while True:
        jobs = await redis_bus.read_jobs(consumer)
        for message_id, run_id in jobs:
            try:
                async with AsyncSessionFactory() as db:
                    await agent_service.process_run(db, run_id)
                await redis_bus.acknowledge_job(message_id)
            except Exception:
                logger.exception("agent run failed run_id=%s", run_id)
        await asyncio.sleep(0)


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()

