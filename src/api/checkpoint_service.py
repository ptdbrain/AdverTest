"""Durable quarantine validation for user-supplied checkpoint bytes."""

from __future__ import annotations

import hashlib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.api.jobs import SqliteRunStore
from src.api.workflow_store import WorkflowJobStore
from src.models.families import FAMILIES


class CheckpointValidationService:
    """Validate metadata and file integrity outside the HTTP request coroutine.

    The quarantine worker deliberately does not deserialize checkpoint pickle
    payloads.  A runnable adapter accepts the artifact later only after a
    deployment-owned model sandbox performs its load/smoke check; local product
    state records that distinction in ``validation_reason``.
    """

    def __init__(self, records: SqliteRunStore, jobs: WorkflowJobStore, *, max_workers: int = 1) -> None:
        self._records = records
        self._jobs = jobs
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="checkpoint-validation")

    def enqueue(self, checkpoint_id: str) -> str:
        job_id = self._jobs.create_job("checkpoint_validation", {"checkpoint_id": checkpoint_id})
        self.start(checkpoint_id, job_id)
        return job_id

    def start(self, checkpoint_id: str, job_id: str) -> None:
        """Start an already persisted validation job after its record is durable."""
        self._executor.submit(self._validate, checkpoint_id, job_id)

    def recover(self) -> None:
        for job in self._jobs.recoverable("checkpoint_validation"):
            checkpoint_id = str(job["request"].get("checkpoint_id", ""))
            if checkpoint_id:
                self._executor.submit(self._validate, checkpoint_id, str(job["id"]))

    def _validate(self, checkpoint_id: str, job_id: str) -> None:
        record = self._records.get_record("checkpoint", checkpoint_id)
        if record is None:
            self._jobs.fail_job(job_id, "CHECKPOINT_UNKNOWN")
            return
        record["status"] = "VALIDATING"
        record["validation"] = {"state": "VALIDATING", "reason": None}
        self._records.update_record("checkpoint", checkpoint_id, record)
        self._jobs.append_event(job_id, "VALIDATING", {"progress_ratio": 0.2, "detail": "Checking quarantined file"})
        try:
            reason = _validate_quarantine_record(record)
        except OSError:
            reason = "CHECKPOINT_UNREADABLE"
        record = self._records.get_record("checkpoint", checkpoint_id) or record
        if reason:
            record["status"] = "REJECTED"
            record["validation_reason"] = reason
            record["validation"] = {"state": "REJECTED", "reason": reason}
            self._records.update_record("checkpoint", checkpoint_id, record)
            self._jobs.fail_job(job_id, reason)
            return
        record["status"] = "READY"
        record["validation_reason"] = None
        record["validation"] = {
            "state": "READY",
            "reason": None,
            "capability_confirmation": "quarantine_integrity_verified",
        }
        self._records.update_record("checkpoint", checkpoint_id, record)
        self._jobs.complete_job(job_id, {"checkpoint_id": checkpoint_id, "status": "READY"})


def _validate_quarantine_record(record: dict[str, Any]) -> str | None:
    path = Path(str(record.get("storage_path", ""))).expanduser().resolve()
    family = FAMILIES.get(str(record.get("family_id", "")))
    if family is None or str(record.get("task_id", "")) not in family.supported_tasks:
        return "MODEL_FAMILY_TASK_MISMATCH"
    if path.suffix.lower() not in family.checkpoint_extensions:
        return "CHECKPOINT_EXTENSION_INVALID"
    if not path.is_file() or path.stat().st_size <= 0:
        return "CHECKPOINT_MISSING"
    if _sha256(path) != record.get("sha256"):
        return "CHECKPOINT_HASH_MISMATCH"
    # PyTorch's current portable checkpoint representation is a zip archive.
    # Reject arbitrary .pt bytes before an adapter or runtime may see them.
    if not zipfile.is_zipfile(path):
        return "CHECKPOINT_FORMAT_INVALID"
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
    if not any(member.endswith("data.pkl") for member in members):
        return "CHECKPOINT_FORMAT_INVALID"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
