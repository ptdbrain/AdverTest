"""Worker-side dispatch for platform job types."""

from __future__ import annotations

from typing import Any

from src.api.checkpoint_service import PlatformCheckpointService
from src.jobs.service import PlatformJobService
from src.storage.export_service import AttackedDatasetExportService


class PlatformWorker:
    """Consumes IDs from a queue and resolves all state through the database."""

    def __init__(
        self,
        jobs: PlatformJobService,
        checkpoints: PlatformCheckpointService,
        exports: AttackedDatasetExportService,
    ) -> None:
        self._jobs = jobs
        self._checkpoints = checkpoints
        self._exports = exports

    def process(self, job_id: str) -> None:
        job = self._jobs.request_for_worker(job_id)
        if job is None or not self._jobs.start(job_id):
            return
        try:
            result = self._run(job)
        except KeyError as exc:
            self._jobs.fail(job_id, str(exc).strip("'"), "Referenced platform record was not found")
            return
        except ValueError as exc:
            self._jobs.fail(job_id, str(exc), str(exc))
            return
        except Exception as exc:
            self._jobs.fail(job_id, "JOB_RUNTIME_FAILED", f"{type(exc).__name__}: {exc}")
            return
        self._jobs.complete(job_id, result)
        validation_job_id = result.get("validation_job_id") if isinstance(result, dict) else None
        if validation_job_id:
            self.process(str(validation_job_id))

    def _run(self, job: dict[str, Any]) -> dict[str, Any]:
        def progress(*, stage: str, completed: int, total: int, message: str) -> bool:
            return self._jobs.progress(job["id"], stage=stage, completed=completed, total=total, message=message)

        if job["type"] == "checkpoint_validation":
            return self._checkpoints.validate_job(job["id"], job["request"], progress)
        if job["type"] == "ultralytics_import":
            return self._checkpoints.import_ultralytics_job(
                job["id"], job["project_id"], job["owner_user_id"], job["request"], progress
            )
        if job["type"] == "attacked_dataset_export":
            progress(stage="EXPORTING", completed=1, total=2, message="Building attacked dataset archive")
            result = self._exports.run(project_id=job["project_id"], actor_id=job["owner_user_id"], request=job["request"])
            progress(stage="PERSISTING", completed=2, total=2, message="Export artifact stored")
            return result
        raise ValueError("JOB_TYPE_UNSUPPORTED")
