"""Render background-worker entry point for the Redis platform queue."""

from __future__ import annotations

from src.api.platform_dependencies import get_platform_queue, get_platform_worker
from src.config import get_settings
from src.jobs.queue import RedisJobQueue


def main() -> None:
    settings = get_settings()
    queue = get_platform_queue()
    if not isinstance(queue, RedisJobQueue):
        raise RuntimeError("platform worker requires QUEUE_BACKEND=redis")
    worker = get_platform_worker()
    timeout = max(1, round(settings.platform_worker_poll_seconds))
    while True:
        job_id = queue.dequeue(timeout)
        if job_id:
            worker.process(job_id)


if __name__ == "__main__":
    main()
