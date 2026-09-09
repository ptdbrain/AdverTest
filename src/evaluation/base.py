"""Task-neutral evaluator boundary consumed by the generic benchmark runner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from src.core.types import ModelPrediction, Sample, Task
from src.evaluation.contracts import FailureCase, MetricEnvelope

if TYPE_CHECKING:
    from src.pipeline.protocol import BenchmarkProtocol


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    task: Task
    protocol_id: str
    headline: MetricEnvelope
    supplemental_metrics: tuple[MetricEnvelope, ...] = ()
    per_sample_metrics: dict[str, tuple[MetricEnvelope, ...]] = Field(default_factory=dict)
    failures: tuple[FailureCase, ...] = ()
    validation_warnings: tuple[str, ...] = ()


class TaskEvaluator(Protocol):
    task: str
    metric_versions: dict[str, str]

    def evaluate(
        self,
        predictions: Sequence[ModelPrediction],
        samples: Sequence[Sample],
        protocol: BenchmarkProtocol,
    ) -> EvaluationResult: ...
