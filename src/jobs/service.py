"""Persistent job state machine with append-only progress events."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select

from src.core.platform_contracts import JobStatus
from src.persistence.database import PlatformDatabase
from src.persistence.models import JobEventRecord, PlatformJobRecord

_TERMINAL = {JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value}


class PlatformJobService:
    """Owns durable job records; queues only carry a job ID and never job state."""

    def __init__(self, database: PlatformDatabase) -> None:
        self._database = database

    def create(
        self,
        *,
        project_id: str,
        owner_user_id: str,
        job_type: str,
        request: dict[str, Any],
        total_units: int = 0,
        idempotency_key: str | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        if idempotency_key:
            with self._database.session() as session:
                existing = session.scalar(
                    select(PlatformJobRecord).where(
                        PlatformJobRecord.project_id == project_id,
                        PlatformJobRecord.idempotency_key == idempotency_key,
                    )
                )
                if existing is not None:
                    return _job_payload(existing)

        job_id = str(uuid4())
        with self._database.session() as session:
            record = PlatformJobRecord(
                id=job_id,
                project_id=project_id,
                owner_user_id=owner_user_id,
                type=job_type,
                status=JobStatus.QUEUED.value,
                stage="QUEUED",
                completed_units=0,
                total_units=total_units,
                idempotency_key=idempotency_key,
                request_json=json.dumps(request, sort_keys=True),
                max_attempts=max_attempts,
            )
            session.add(record)
            session.flush()
            self._append_event(session, record, "QUEUED", 0, total_units, "Job queued")
            return _job_payload(record)

    def get(self, project_id: str, job_id: str) -> dict[str, Any] | None:
        with self._database.session() as session:
            record = session.scalar(
                select(PlatformJobRecord).where(
                    PlatformJobRecord.project_id == project_id, PlatformJobRecord.id == job_id
                )
            )
            return _job_payload(record) if record else None

    def list(self, project_id: str, *, job_type: str | None = None) -> list[dict[str, Any]]:
        """List jobs for one project without exposing records from another."""
        with self._database.session() as session:
            statement = select(PlatformJobRecord).where(PlatformJobRecord.project_id == project_id)
            if job_type is not None:
                statement = statement.where(PlatformJobRecord.type == job_type)
            records = session.scalars(statement.order_by(PlatformJobRecord.created_at.desc())).all()
            return [_job_payload(record) for record in records]

    def cancel_requested(self, job_id: str) -> bool:
        with self._database.session() as session:
            record = session.get(PlatformJobRecord, job_id)
            return bool(record and record.cancel_requested)

    def request_for_worker(self, job_id: str) -> dict[str, Any] | None:
        with self._database.session() as session:
            record = session.get(PlatformJobRecord, job_id)
            if record is None:
                return None
            return _job_payload(record)

    def events(self, project_id: str, job_id: str, *, after_sequence: int = -1) -> list[dict[str, Any]]:
        with self._database.session() as session:
            exists = session.scalar(
                select(PlatformJobRecord.id).where(
                    PlatformJobRecord.project_id == project_id, PlatformJobRecord.id == job_id
                )
            )
            if exists is None:
                return []
            records = session.scalars(
                select(JobEventRecord)
                .where(JobEventRecord.job_id == job_id, JobEventRecord.sequence > after_sequence)
                .order_by(JobEventRecord.sequence)
            ).all()
            return [_event_payload(record) for record in records]

    def start(self, job_id: str, message: str = "Worker started") -> bool:
        with self._database.session() as session:
            record = session.get(PlatformJobRecord, job_id)
            # A queue can deliver the same message more than once.  Only the
            # first worker may claim a queued job; accepting RUNNING here
            # would execute it concurrently and append conflicting events.
            if record is None or record.status != JobStatus.QUEUED.value or record.cancel_requested:
                return False
            record.status = JobStatus.RUNNING.value
            record.stage = "RUNNING"
            record.started_at = record.started_at or datetime.now(UTC)
            record.attempt += 1
            self._append_event(session, record, "RUNNING", record.completed_units, record.total_units, message)
            return True

    def waiting_for_gpu(self, job_id: str, message: str = "GPU is starting") -> bool:
        """Expose GPU cold-start progress without letting a worker claim the job.

        The durable job remains ``QUEUED`` so that :meth:`start` is still the
        only transition that grants execution ownership to the Pub/Sub worker.
        """
        with self._database.session() as session:
            record = session.get(PlatformJobRecord, job_id)
            if record is None or record.status != JobStatus.QUEUED.value or record.cancel_requested:
                return False
            record.stage = "GPU_STARTING"
            self._append_event(session, record, "GPU_STARTING", record.completed_units, record.total_units, message)
            return True

    def progress(self, job_id: str, *, stage: str, completed: int, total: int, message: str) -> bool:
        if completed < 0 or total < 0 or (total and completed > total):
            raise ValueError("invalid job progress units")
        with self._database.session() as session:
            record = session.get(PlatformJobRecord, job_id)
            if record is None or record.status in _TERMINAL or record.cancel_requested:
                return False
            record.status = JobStatus.RUNNING.value
            record.stage = stage
            record.completed_units = completed
            record.total_units = total
            self._append_event(session, record, stage, completed, total, message)
            return True

    def complete(self, job_id: str, result: dict[str, Any], message: str = "Job completed") -> bool:
        with self._database.session() as session:
            record = session.get(PlatformJobRecord, job_id)
            if record is None or record.status in _TERMINAL:
                return False
            if record.cancel_requested:
                return self._finish(
                    session, record, JobStatus.CANCELLED.value, "JOB_CANCELLED", "Cancellation requested"
                )
            record.status = JobStatus.COMPLETED.value
            record.stage = "COMPLETED"
            record.completed_units = record.total_units or record.completed_units
            record.result_json = json.dumps(result, sort_keys=True)
            record.completed_at = datetime.now(UTC)
            self._append_event(session, record, "COMPLETED", record.completed_units, record.total_units, message)
            return True

    def fail(self, job_id: str, error_code: str, error_message: str) -> bool:
        with self._database.session() as session:
            record = session.get(PlatformJobRecord, job_id)
            if record is None or record.status in _TERMINAL:
                return False
            status = JobStatus.CANCELLED.value if record.cancel_requested else JobStatus.FAILED.value
            return self._finish(session, record, status, error_code, error_message)

    def cancel(self, project_id: str, job_id: str) -> bool:
        with self._database.session() as session:
            record = session.scalar(
                select(PlatformJobRecord).where(
                    PlatformJobRecord.project_id == project_id, PlatformJobRecord.id == job_id
                )
            )
            if record is None:
                return False
            if record.status in _TERMINAL:
                return True
            record.cancel_requested = True
            self._append_event(
                session,
                record,
                "CANCEL_REQUESTED",
                record.completed_units,
                record.total_units,
                "Cancellation requested",
            )
            return True

    def retry(self, project_id: str, job_id: str) -> dict[str, Any] | None:
        with self._database.session() as session:
            record = session.scalar(
                select(PlatformJobRecord).where(
                    PlatformJobRecord.project_id == project_id, PlatformJobRecord.id == job_id
                )
            )
            if record is None or record.status not in {JobStatus.FAILED.value, JobStatus.CANCELLED.value}:
                return None
            if record.attempt >= record.max_attempts:
                return None
            record.status = JobStatus.QUEUED.value
            record.stage = "QUEUED"
            record.error_code = None
            record.error_message = None
            record.cancel_requested = False
            record.completed_at = None
            self._append_event(
                session, record, "QUEUED", record.completed_units, record.total_units, "Job retry queued"
            )
            return _job_payload(record)

    @staticmethod
    def _append_event(
        session: Any, record: PlatformJobRecord, stage: str, completed: int, total: int, message: str
    ) -> None:
        sequence = (
            session.scalar(
                select(func.coalesce(func.max(JobEventRecord.sequence), -1)).where(JobEventRecord.job_id == record.id)
            )
            + 1
        )
        percent = 0.0 if total == 0 else round(100.0 * completed / total, 4)
        session.add(
            JobEventRecord(
                job_id=record.id,
                sequence=sequence,
                stage=stage,
                completed=completed,
                total=total,
                percent=percent,
                message=message,
            )
        )

    def _finish(
        self, session: Any, record: PlatformJobRecord, status: str, error_code: str, error_message: str
    ) -> bool:
        record.status = status
        record.stage = status
        record.error_code = error_code
        record.error_message = error_message
        record.completed_at = datetime.now(UTC)
        self._append_event(session, record, status, record.completed_units, record.total_units, error_message)
        return True


def _job_payload(record: PlatformJobRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "project_id": record.project_id,
        "owner_user_id": record.owner_user_id,
        "type": record.type,
        "status": record.status,
        "stage": record.stage,
        "completed_units": record.completed_units,
        "total_units": record.total_units,
        "idempotency_key": record.idempotency_key,
        "request": json.loads(record.request_json),
        "result": json.loads(record.result_json) if record.result_json else None,
        "error_code": record.error_code,
        "error_message": record.error_message,
        "cancel_requested": record.cancel_requested,
        "attempt": record.attempt,
        "max_attempts": record.max_attempts,
        "created_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
    }


def _event_payload(record: JobEventRecord) -> dict[str, Any]:
    return {
        "job_id": record.job_id,
        "sequence": record.sequence,
        "stage": record.stage,
        "completed": record.completed,
        "total": record.total,
        "percent": record.percent,
        "message": record.message,
        "timestamp": record.created_at,
    }
