"""Guarded synchronous worker orchestration for local or queued execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.core.events import ProgressEvent
from src.core.hashing import file_digest
from src.training.base import TrainerCallbacks
from src.training.contracts import TrainingRunConfig
from src.training.registry import TrainerRegistry
from src.training.report import TrainingReport, TrainingState, TrainingStateMachine


class ComputeWorker:
    def __init__(
        self,
        registry: TrainerRegistry,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.cancel_check = cancel_check or (lambda: False)
        self._sequence = 0

    def run(
        self,
        config: TrainingRunConfig,
        *,
        callbacks: Callable[[ProgressEvent], None]
        | Sequence[Callable[[ProgressEvent], None]]
        | None = None,
    ) -> TrainingReport:
        emitters = self._callbacks(callbacks)
        machine = TrainingStateMachine()
        trainer = self.registry.get(config.trainer_name)
        report = TrainingReport(run_id=config.run_id, state="DRAFT")
        try:
            self._move(machine, "VALIDATING", config, emitters)
            validation = trainer.validate_config(config)
            if not validation.valid:
                return self._terminal(machine, "FAILED", config, emitters, errors=validation.errors)
            cancelled = self._cancelled(machine, config, emitters)
            if cancelled:
                return cancelled

            self._move(machine, "ESTIMATING", config, emitters)
            estimate = trainer.estimate(config)
            report = report.model_copy(update={"state": machine.state, "estimate": estimate})
            exceeded = self._budget_exceeded(config, estimate)
            if exceeded:
                return self._terminal(
                    machine,
                    "BUDGET_EXCEEDED",
                    config,
                    emitters,
                    errors=tuple(exceeded),
                    report=report,
                )
            cancelled = self._cancelled(machine, config, emitters, report=report)
            if cancelled:
                return cancelled

            self._move(machine, "QUEUED", config, emitters)
            self._move(machine, "PREPARING_DATA", config, emitters)
            prepared = trainer.prepare_data(config)
            report = report.model_copy(update={"state": machine.state, "prepared_data": prepared})
            if not prepared.lineage_valid:
                return self._terminal(
                    machine,
                    "FAILED",
                    config,
                    emitters,
                    errors=("training data lineage/leakage validation failed",),
                    report=report,
                )

            self._move(machine, "TRAINING", config, emitters)
            epoch_metrics: list[dict[str, float]] = []

            def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
                epoch_metrics.append({"epoch": float(epoch), **metrics})
                self._emit(machine.state, config, emitters, {"epoch": epoch, "metrics": metrics})

            trained = trainer.train(
                config,
                TrainerCallbacks(on_epoch=on_epoch, is_cancelled=self.cancel_check),
            )
            report = trained.model_copy(
                update={
                    "state": machine.state,
                    "estimate": estimate,
                    "prepared_data": prepared,
                    "epoch_metrics": tuple(epoch_metrics) or trained.epoch_metrics,
                }
            )
            cancelled = self._cancelled(machine, config, emitters, report=report)
            if cancelled:
                return cancelled
            if report.checkpoint is None:
                return self._terminal(
                    machine, "FAILED", config, emitters, errors=("trainer returned no checkpoint",), report=report
                )

            self._move(machine, "VALIDATING_CHECKPOINT", config, emitters)
            checkpoint = Path(report.checkpoint.path).expanduser().resolve()
            if not checkpoint.is_file() or file_digest(checkpoint, length=64) != report.checkpoint.sha256:
                return self._terminal(
                    machine,
                    "FAILED",
                    config,
                    emitters,
                    errors=("checkpoint hash validation failed",),
                    report=report,
                )
            snapshot = trainer.evaluate_checkpoint(report.checkpoint)
            report = report.model_copy(update={"state": machine.state, "checkpoint_metrics": snapshot})

            self._move(machine, "EXPORTING", config, emitters)
            exported = trainer.export_checkpoint(report.checkpoint)
            if not exported.load_valid or exported.sha256 != report.checkpoint.sha256:
                return self._terminal(
                    machine,
                    "FAILED",
                    config,
                    emitters,
                    errors=("exported checkpoint failed load/hash validation",),
                    report=report,
                )
            report = report.model_copy(update={"state": machine.state, "exported_checkpoint": exported})

            self._move(machine, "REGISTERING_MODEL", config, emitters)
            registration = {
                "model_version": f"{config.model_version}+{config.run_id}",
                "parent_model_version": config.model_version,
                "trainer": trainer.metadata().model_dump(mode="json"),
                "training_manifest_id": prepared.manifest_id,
                "training_manifest_hash": prepared.manifest_hash,
                "checkpoint_hash": exported.sha256,
            }
            report = report.model_copy(update={"state": machine.state, "registration": registration})
            self._move(machine, "COMPLETED", config, emitters)
            return report.model_copy(update={"state": "COMPLETED"})
        except Exception as exc:
            if machine.state in {"COMPLETED", "FAILED", "CANCELLED", "BUDGET_EXCEEDED"}:
                raise
            return self._terminal(
                machine,
                "FAILED",
                config,
                emitters,
                errors=(f"{type(exc).__name__}: {exc}",),
                report=report,
            )

    def _move(self, machine, state: TrainingState, config, emitters) -> None:
        machine.transition(state)
        self._emit(state, config, emitters)

    def _terminal(self, machine, state, config, emitters, *, errors=(), report=None):
        machine.transition(state)
        self._emit(state, config, emitters, {"errors": errors})
        current = report or TrainingReport(run_id=config.run_id, state=state)
        return current.model_copy(update={"state": state, "errors": tuple(errors)})

    def _cancelled(self, machine, config, emitters, *, report=None):
        if not self.cancel_check():
            return None
        return self._terminal(
            machine,
            "CANCELLED",
            config,
            emitters,
            errors=("cancellation requested",),
            report=report,
        )

    @staticmethod
    def _budget_exceeded(config, estimate) -> list[str]:
        reasons = []
        if config.max_gpu_hours is not None and estimate.gpu_hours > config.max_gpu_hours:
            reasons.append("gpu hours estimate exceeds cap")
        if config.max_storage_bytes is not None and estimate.storage_bytes > config.max_storage_bytes:
            reasons.append("storage estimate exceeds cap")
        if config.max_wall_time_seconds is not None and estimate.wall_time_seconds > config.max_wall_time_seconds:
            reasons.append("wall time estimate exceeds cap")
        return reasons

    @staticmethod
    def _callbacks(callbacks):
        if callbacks is None:
            return ()
        return (callbacks,) if callable(callbacks) else tuple(callbacks)

    def _emit(self, state, config, callbacks, detail=None):
        event = ProgressEvent(
            job_id=config.run_id,
            job_type="training",
            state=state,
            progress_ratio=min(1.0, self._sequence / 10.0),
            sequence=self._sequence,
            detail=detail or {},
            created_at=datetime.now(UTC),
        )
        self._sequence += 1
        for callback in callbacks:
            callback(event)
