"""Explicit compute backend boundary; queue semantics stay backend-neutral."""

from __future__ import annotations

from typing import Any, Protocol

from src.jobs.queue import JobQueue


class ComputeBackend(Protocol):
    def dispatch(self, job_id: str) -> dict[str, Any] | None: ...


class LocalWorker:
    def __init__(self, queue: JobQueue) -> None:
        self._queue = queue

    def dispatch(self, job_id: str) -> None:
        self._queue.enqueue(job_id)
        return None


class RenderWorker(LocalWorker):
    """Render uses the same Redis queue; a separate background service consumes it."""


class ExternalGPUWorker:
    """Dispatches a job ID to an external GPU service without exposing payloads."""

    def __init__(self, queue: JobQueue) -> None:
        self._queue = queue

    def dispatch(self, job_id: str) -> None:
        self._queue.enqueue(job_id)
