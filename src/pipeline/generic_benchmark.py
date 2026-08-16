"""Task-neutral, protocol-locked benchmark orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.adapters.base import ModelAdapter
from src.core.events import ProgressEvent
from src.core.hashing import stable_digest
from src.core.types import (
    DetectionPrediction,
    ModelInfo,
    ModelPrediction,
    Sample,
    SegmentationPrediction,
    Task,
)
from src.evaluation.base import EvaluationResult, TaskEvaluator
from src.pipeline.protocol import BenchmarkProtocol

ProgressCallback = Callable[[ProgressEvent], None]
SampleProvider = Callable[[BenchmarkProtocol], Sequence[Sample]]
VariantProvider = Callable[[str, Sequence[Sample]], Sequence[Sample]]
CancelCheck = Callable[[], bool]


class _FrozenReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class GenericCellResult(_FrozenReport):
    recipe_hash: str
    evaluation: EvaluationResult
    sample_ids: tuple[str, ...]


class GenericModelResult(_FrozenReport):
    model_id: str
    model_version: str
    task: Task
    clean: EvaluationResult
    cells: tuple[GenericCellResult, ...]
    paired_sample_ids: tuple[str, ...]


class ModelIncompatibility(_FrozenReport):
    model_id: str
    task: str
    reason: str


class GenericBenchmarkReport(_FrozenReport):
    protocol_id: str
    models: tuple[GenericModelResult, ...]
    skipped: tuple[ModelIncompatibility, ...] = ()
    resumed_cells: int = 0
    validation_warnings: tuple[str, ...] = ()
    complete: bool = True
    cancellation_reason: str | None = None


class BenchmarkRunner:
    """Evaluate multiple task adapters against one immutable protocol.

    The injected providers keep dataset and recipe materialization outside the
    runner. ``checkpoint_store`` may be retained by a worker between calls; its
    keys include complete protocol/model/task/recipe identity.
    """

    def __init__(
        self,
        *,
        sample_provider: SampleProvider,
        variant_provider: VariantProvider,
        checkpoint_store: dict[str, EvaluationResult] | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> None:
        self._sample_provider = sample_provider
        self._variant_provider = variant_provider
        self._checkpoints = checkpoint_store if checkpoint_store is not None else {}
        self._cancel_check = cancel_check or (lambda: False)
        self._sequence = 0

    def run(
        self,
        protocol: BenchmarkProtocol,
        models: Sequence[ModelAdapter],
        evaluator_registry: Mapping[str, TaskEvaluator],
        callbacks: ProgressCallback | Sequence[ProgressCallback] | None = None,
    ) -> GenericBenchmarkReport:
        if protocol.status != "LOCKED":
            raise ValueError("benchmark protocol must be LOCKED before execution")

        clean_samples = tuple(self._sample_provider(protocol))
        self._validate_samples(clean_samples, protocol.sample_ids, context="clean")
        variants = {
            recipe_hash: tuple(self._variant_provider(recipe_hash, clean_samples))
            for recipe_hash in protocol.recipe_hashes
        }
        for recipe_hash, samples in variants.items():
            self._validate_samples(samples, protocol.sample_ids, context=f"recipe {recipe_hash}")

        emitters = self._callbacks(callbacks)
        total_cells = max(1, len(models) * (1 + len(protocol.recipe_hashes)))
        completed = 0
        resumed_cells = 0
        results: list[GenericModelResult] = []
        skipped: list[ModelIncompatibility] = []
        cancelled = False

        for model in models:
            if self._cancel_check():
                cancelled = True
                break
            info = model.metadata()
            evaluator = evaluator_registry.get(info.task)
            incompatibility = self._incompatibility(info, evaluator, protocol)
            if incompatibility is not None:
                skipped.append(incompatibility)
                completed += 1 + len(protocol.recipe_hashes)
                self._emit(
                    emitters,
                    protocol,
                    "SKIPPED",
                    min(1.0, completed / total_cells),
                    {"model": info.name, "reason": incompatibility.reason},
                )
                continue
            assert evaluator is not None

            clean_key = self._checkpoint_key(protocol, info, "clean")
            clean = self._checkpoints.get(clean_key)
            if clean is None:
                predictions = model.predict(clean_samples)
                self._validate_predictions(predictions, protocol.sample_ids, info.task)
                clean = evaluator.evaluate(predictions, clean_samples, protocol)
                self._validate_evaluation(clean, info.task, protocol)
                self._checkpoints[clean_key] = clean
            completed += 1
            self._emit(
                emitters,
                protocol,
                "CLEAN_EVALUATED",
                min(1.0, completed / total_cells),
                {"model": info.name, "task": info.task},
            )

            cells: list[GenericCellResult] = []
            for recipe_hash in protocol.recipe_hashes:
                if self._cancel_check():
                    cancelled = True
                    break
                cell_key = self._checkpoint_key(protocol, info, recipe_hash)
                evaluation = self._checkpoints.get(cell_key)
                resumed = evaluation is not None
                if evaluation is None:
                    samples = variants[recipe_hash]
                    predictions = model.predict(samples)
                    self._validate_predictions(predictions, protocol.sample_ids, info.task)
                    evaluation = evaluator.evaluate(predictions, samples, protocol)
                    self._validate_evaluation(evaluation, info.task, protocol)
                    self._checkpoints[cell_key] = evaluation
                else:
                    resumed_cells += 1
                cells.append(
                    GenericCellResult(
                        recipe_hash=recipe_hash,
                        evaluation=evaluation,
                        sample_ids=protocol.sample_ids,
                    )
                )
                completed += 1
                self._emit(
                    emitters,
                    protocol,
                    "CELL_RESUMED" if resumed else "CELL_EVALUATED",
                    min(1.0, completed / total_cells),
                    {"model": info.name, "task": info.task, "recipe_hash": recipe_hash},
                )

            results.append(
                GenericModelResult(
                    model_id=info.name,
                    model_version=info.version,
                    task=info.task,
                    clean=clean,
                    cells=tuple(cells),
                    paired_sample_ids=protocol.sample_ids,
                )
            )
            if cancelled:
                break

        return GenericBenchmarkReport(
            protocol_id=protocol.protocol_id,
            models=tuple(results),
            skipped=tuple(skipped),
            resumed_cells=resumed_cells,
            complete=not cancelled,
            cancellation_reason="cancel_check requested cancellation" if cancelled else None,
        )

    @staticmethod
    def _validate_samples(
        samples: Sequence[Sample],
        expected_ids: tuple[str, ...],
        *,
        context: str,
    ) -> None:
        sample_ids = tuple(sample.sample_id for sample in samples)
        if sample_ids != expected_ids:
            raise ValueError(
                f"{context} sample IDs/order do not match locked protocol: "
                f"expected {expected_ids!r}, got {sample_ids!r}"
            )

    @staticmethod
    def _validate_predictions(
        predictions: Sequence[ModelPrediction],
        expected_ids: tuple[str, ...],
        task: Task,
    ) -> None:
        prediction_ids = tuple(prediction.sample_id for prediction in predictions)
        if prediction_ids != expected_ids:
            raise ValueError(
                "prediction sample IDs/order do not match locked protocol: "
                f"expected {expected_ids!r}, got {prediction_ids!r}"
            )
        expected_type = SegmentationPrediction if task == "segmentation" else DetectionPrediction
        wrong = [type(item).__name__ for item in predictions if not isinstance(item, expected_type)]
        if wrong:
            raise TypeError(f"task {task!r} returned incompatible predictions: {wrong!r}")

    @staticmethod
    def _validate_evaluation(
        evaluation: EvaluationResult,
        task: Task,
        protocol: BenchmarkProtocol,
    ) -> None:
        if evaluation.task != task:
            raise ValueError(
                f"evaluator returned task {evaluation.task!r} for model task {task!r}"
            )
        if evaluation.protocol_id != protocol.protocol_id:
            raise ValueError("evaluator result does not reference the locked protocol")
        if set(evaluation.per_sample_metrics) - set(protocol.sample_ids):
            raise ValueError("evaluator returned metrics for samples outside the protocol")

    @staticmethod
    def _incompatibility(
        info: ModelInfo,
        evaluator: TaskEvaluator | None,
        protocol: BenchmarkProtocol,
    ) -> ModelIncompatibility | None:
        if not info.runnable:
            return ModelIncompatibility(
                model_id=info.name,
                task=info.task,
                reason="model metadata marks adapter as not runnable",
            )
        if evaluator is None:
            return ModelIncompatibility(
                model_id=info.name,
                task=info.task,
                reason=f"no evaluator registered for task {info.task!r}",
            )
        mismatches = {
            name: (version, protocol.metric_versions.get(name))
            for name, version in evaluator.metric_versions.items()
            if protocol.metric_versions.get(name) != version
        }
        if mismatches:
            return ModelIncompatibility(
                model_id=info.name,
                task=info.task,
                reason=f"metric version mismatch: {mismatches!r}",
            )
        return None

    @staticmethod
    def _checkpoint_key(
        protocol: BenchmarkProtocol,
        info: ModelInfo,
        recipe_hash: str,
    ) -> str:
        return stable_digest(
            {
                "cache_type": "benchmark-evaluation",
                "protocol_id": protocol.protocol_id,
                "model": info.name,
                "model_version": info.version,
                "checkpoint_hash": info.checkpoint_hash,
                "preprocessing_version": info.preprocessing_version,
                "task": info.task,
                "recipe_hash": recipe_hash,
            },
            length=64,
        )

    @staticmethod
    def _callbacks(
        callbacks: ProgressCallback | Sequence[ProgressCallback] | None,
    ) -> tuple[ProgressCallback, ...]:
        if callbacks is None:
            return ()
        if callable(callbacks):
            return (callbacks,)
        return tuple(callbacks)

    def _emit(
        self,
        callbacks: Sequence[ProgressCallback],
        protocol: BenchmarkProtocol,
        state: str,
        progress_ratio: float,
        detail: dict[str, Any],
    ) -> None:
        event = ProgressEvent(
            job_id=protocol.protocol_id,
            job_type="benchmark",
            state=state,
            progress_ratio=progress_ratio,
            sequence=self._sequence,
            detail=detail,
            created_at=datetime.now(UTC),
        )
        self._sequence += 1
        for callback in callbacks:
            callback(event)
