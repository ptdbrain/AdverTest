"""Owner-neutral trainer plugin boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from src.training.contracts import TrainingRunConfig
from src.training.report import (
    CheckpointMetadata,
    ExportedCheckpoint,
    MetricSnapshot,
    PreparedTrainingData,
    TrainerMetadata,
    TrainingEstimate,
    TrainingReport,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class TrainerCallbacks:
    on_epoch: Callable[[int, dict[str, float]], None]
    is_cancelled: Callable[[], bool] = lambda: False


class ModelTrainer(ABC):
    @abstractmethod
    def validate_config(self, config: TrainingRunConfig) -> ValidationReport: ...

    @abstractmethod
    def estimate(self, config: TrainingRunConfig) -> TrainingEstimate: ...

    @abstractmethod
    def prepare_data(self, config: TrainingRunConfig) -> PreparedTrainingData: ...

    @abstractmethod
    def train(
        self,
        config: TrainingRunConfig,
        callbacks: TrainerCallbacks,
    ) -> TrainingReport: ...

    @abstractmethod
    def evaluate_checkpoint(self, checkpoint: CheckpointMetadata) -> MetricSnapshot: ...

    @abstractmethod
    def export_checkpoint(self, checkpoint: CheckpointMetadata) -> ExportedCheckpoint: ...

    @abstractmethod
    def metadata(self) -> TrainerMetadata: ...
