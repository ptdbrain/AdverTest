"""Offline training services for reusable attack artifacts."""

from src.training.base import ModelTrainer, TrainerCallbacks
from src.training.contracts import DefenseProfile, TrainingRunConfig
from src.training.dataset_builder import (
    TrainingDatasetBuilder,
    TrainingDatasetConfig,
    TrainingDatasetManifest,
)
from src.training.patch_trainer import PatchArtifact, PatchTrainer, PatchTrainingConfig
from src.training.registry import TrainerRegistry
from src.training.report import (
    CheckpointMetadata,
    ExportedCheckpoint,
    MetricSnapshot,
    PreparedTrainingData,
    TrainerMetadata,
    TrainingEstimate,
    TrainingReport,
    TrainingStateMachine,
    ValidationReport,
)
from src.training.worker import ComputeWorker

__all__ = [
    "PatchArtifact",
    "PatchTrainer",
    "PatchTrainingConfig",
    "TrainingDatasetBuilder",
    "TrainingDatasetConfig",
    "TrainingDatasetManifest",
    "CheckpointMetadata",
    "ComputeWorker",
    "DefenseProfile",
    "ExportedCheckpoint",
    "MetricSnapshot",
    "ModelTrainer",
    "PreparedTrainingData",
    "TrainerCallbacks",
    "TrainerMetadata",
    "TrainerRegistry",
    "TrainingEstimate",
    "TrainingReport",
    "TrainingRunConfig",
    "TrainingStateMachine",
    "ValidationReport",
]
