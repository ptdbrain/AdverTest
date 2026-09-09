"""Immutable reports exchanged by trainers and compute workers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TrainingState = Literal[
    "DRAFT",
    "VALIDATING",
    "ESTIMATING",
    "QUEUED",
    "PREPARING_DATA",
    "TRAINING",
    "VALIDATING_CHECKPOINT",
    "EXPORTING",
    "REGISTERING_MODEL",
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "BUDGET_EXCEEDED",
]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ValidationReport(_Frozen):
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class TrainingEstimate(_Frozen):
    gpu_hours: float = Field(ge=0.0)
    storage_bytes: int = Field(ge=0)
    wall_time_seconds: int = Field(ge=0)


class PreparedTrainingData(_Frozen):
    manifest_id: str
    manifest_hash: str
    lineage_valid: bool
    leakage_report_id: str | None = None


class CheckpointMetadata(_Frozen):
    path: str
    sha256: str
    parent_model_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricSnapshot(_Frozen):
    metrics: dict[str, float]
    version: str = "1.0.0"


class ExportedCheckpoint(_Frozen):
    path: str
    sha256: str
    load_valid: bool


class TrainerMetadata(_Frozen):
    name: str
    task: Literal["detection2d", "segmentation", "detection3d"]
    version: str


class TrainingReport(_Frozen):
    run_id: str
    state: TrainingState
    estimate: TrainingEstimate | None = None
    prepared_data: PreparedTrainingData | None = None
    epoch_metrics: tuple[dict[str, float], ...] = ()
    checkpoint: CheckpointMetadata | None = None
    checkpoint_metrics: MetricSnapshot | None = None
    exported_checkpoint: ExportedCheckpoint | None = None
    registration: dict[str, Any] | None = None
    resume_metadata: dict[str, Any] = Field(default_factory=dict)
    errors: tuple[str, ...] = ()


class TrainingStateMachine:
    """Small explicit lifecycle; terminal states cannot transition further."""

    _next = {
        "DRAFT": {"VALIDATING"},
        "VALIDATING": {"ESTIMATING", "FAILED", "CANCELLED"},
        "ESTIMATING": {"QUEUED", "FAILED", "CANCELLED", "BUDGET_EXCEEDED"},
        "QUEUED": {"PREPARING_DATA", "FAILED", "CANCELLED"},
        "PREPARING_DATA": {"TRAINING", "FAILED", "CANCELLED"},
        "TRAINING": {"VALIDATING_CHECKPOINT", "FAILED", "CANCELLED", "BUDGET_EXCEEDED"},
        "VALIDATING_CHECKPOINT": {"EXPORTING", "FAILED", "CANCELLED"},
        "EXPORTING": {"REGISTERING_MODEL", "FAILED", "CANCELLED"},
        "REGISTERING_MODEL": {"COMPLETED", "FAILED", "CANCELLED"},
        "COMPLETED": set(),
        "FAILED": set(),
        "CANCELLED": set(),
        "BUDGET_EXCEEDED": set(),
    }

    def __init__(self) -> None:
        self.state: TrainingState = "DRAFT"

    def transition(self, target: TrainingState) -> TrainingState:
        if target not in self._next[self.state]:
            raise ValueError(f"illegal training transition {self.state} -> {target}")
        self.state = target
        return self.state
