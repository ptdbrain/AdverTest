"""Durable bridge between HTTP training jobs and the trainer worker."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.api.workflow_store import WorkflowJobStore
from src.core.events import ProgressEvent
from src.training.contracts import TrainingRunConfig
from src.training.registry import TrainerRegistry
from src.training.worker import ComputeWorker


class TrainingJobService:
    """Run registered trainers outside HTTP handlers and persist their evidence."""

    def __init__(self, store: WorkflowJobStore, registry: TrainerRegistry, *, max_workers: int = 1) -> None:
        self.store = store
        self.registry = registry
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="advertest-training")

    def enqueue(self, request: TrainingRunConfig) -> str:
        job_id = self.store.create_job("training", request.model_dump(mode="json"))
        config = request.model_copy(update={"run_id": job_id})
        self.pool.submit(self._execute, job_id, config)
        return job_id

    def get(self, job_id: str) -> dict | None:
        return self.store.get_job(job_id)

    def list(self) -> list[dict]:
        return self.store.jobs("training")

    def cancel(self, job_id: str) -> bool:
        return self.store.request_cancel(job_id)

    def estimate(self, request: TrainingRunConfig) -> dict:
        trainer = self.registry.get(request.trainer_name)
        return trainer.estimate(request).model_dump(mode="json")

    def recover(self) -> list[str]:
        recovered: list[str] = []
        for job in self.store.recoverable("training"):
            config = TrainingRunConfig(**job["request"]).model_copy(update={"run_id": job["id"]})
            self.pool.submit(self._execute, job["id"], config)
            recovered.append(job["id"])
        return recovered

    def _execute(self, job_id: str, config: TrainingRunConfig) -> None:
        worker = ComputeWorker(self.registry, cancel_check=lambda: self.store.cancel_requested(job_id))

        def persist_event(event: ProgressEvent) -> None:
            self.store.append_event(
                job_id,
                event.state,
                {"progress_ratio": event.progress_ratio, "detail": event.detail, "sequence": event.sequence},
                update_status=event.state not in {"COMPLETED", "FAILED", "CANCELLED"},
            )

        try:
            report = worker.run(config, callbacks=persist_event)
        except Exception as exc:
            self.store.fail_job(job_id, f"{type(exc).__name__}: {exc}")
            return
        if report.state == "COMPLETED" and report.checkpoint and report.registration:
            result = report.model_dump(mode="json")
            result["model_version"] = {
                "id": report.registration["model_version"],
                "parent_id": report.registration["parent_model_version"],
                "checkpoint_hash": report.checkpoint.sha256,
            }
            self.store.complete_job(job_id, result)
            return
        errors = "; ".join(report.errors) or f"training ended in {report.state}"
        self.store.fail_job(job_id, errors, cancelled=report.state == "CANCELLED")
