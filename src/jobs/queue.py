"""Queue transports. Durable state remains in :mod:`src.jobs.service`."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol


class JobQueue(Protocol):
    def enqueue(self, job_id: str) -> None: ...


class LocalJobQueue:
    """Development queue that invokes the consumer in a bounded thread pool."""

    def __init__(self, consumer: Callable[[str], None], max_workers: int) -> None:
        self._consumer = consumer
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="advertest-platform")

    def enqueue(self, job_id: str) -> None:
        self._executor.submit(self._consumer, job_id)


class RedisJobQueue:
    """Redis list queue shared by Render web and worker services."""

    def __init__(self, redis_url: str, queue_name: str = "advertest-platform-jobs") -> None:
        from redis import Redis

        self._client = Redis.from_url(redis_url, decode_responses=True)
        self._queue_name = queue_name

    def enqueue(self, job_id: str) -> None:
        self._client.lpush(self._queue_name, job_id)

    def dequeue(self, timeout_seconds: int) -> str | None:
        item = self._client.brpop(self._queue_name, timeout=timeout_seconds)
        return item[1] if item else None
