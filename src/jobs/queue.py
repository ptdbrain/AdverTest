"""Queue transports. Durable state remains in :mod:`src.jobs.service`."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


class HttpDispatcherQueue:
    """Send job IDs to a GCP dispatcher; the dispatcher publishes Pub/Sub."""

    def __init__(self, url: str, token: str | None = None) -> None:
        self._url = url
        self._token = token

    def enqueue(self, job_id: str) -> None:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(
            self._url,
            data=json.dumps({"job_id": job_id}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=10) as response:  # noqa: S310 - deployment configuration
                if not 200 <= response.status < 300:
                    raise RuntimeError(f"external dispatcher returned HTTP {response.status}")
        except (HTTPError, URLError) as exc:
            raise RuntimeError("EXTERNAL_QUEUE_DISPATCH_FAILED") from exc
