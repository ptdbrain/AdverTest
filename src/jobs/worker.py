"""Worker-side dispatch for platform job types."""

from __future__ import annotations

from typing import Any

from src.api.checkpoint_service import PlatformCheckpointService
from src.config import get_settings
from src.jobs.service import PlatformJobService
from src.pipeline.runner import RunConfig, TestRunner
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
        self._runner = TestRunner()

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
        if job["type"] == "benchmark_run":
            config = RunConfig.model_validate(job["request"])
            settings = get_settings()
            if config.model == "yolo11":
                adapter_params = dict(config.adapter_params)
                adapter_params.update(
                    {
                        "device": settings.model_device,
                        "half": settings.model_half_precision,
                        "batch_size": settings.model_batch_size,
                    }
                )
                config = config.model_copy(update={"adapter_params": adapter_params})

            estimate = self._runner.estimate(config)
            total = max(1, estimate.n_cells)

            def run_progress(stage: str, detail: dict[str, Any]) -> None:
                completed = int(detail.get("completed_cells", 0))
                progress(
                    stage=stage,
                    completed=min(completed, total),
                    total=total,
                    message=str(detail.get("attack") or detail.get("phase") or stage),
                )

            report = self._runner.run(
                config,
                progress=run_progress,
                should_cancel=lambda: self._jobs.cancel_requested(job["id"]),
                run_id=job["id"],
            )
            return report.as_dict()
        raise ValueError("JOB_TYPE_UNSUPPORTED")
