"""Wave 2 — Closed-loop failure-to-retraining chain.

Full chain: FailureCase → FailureCluster → RetrainingBacklog → DefenseProfile →
TrainingDatasetManifest → TrainingRun → checkpoint → checkpoint gate →
child ModelVersion → locked re-benchmark → RecoveryReport

This module ties together the individual contracts from evaluation, training and
models into a single auditable pipeline with lineage tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.hashing import stable_digest

ClosedLoopState = Literal[
    "FAILURE_IDENTIFIED",
    "CLUSTER_FORMED",
    "BACKLOG_CREATED",
    "BACKLOG_APPROVED",
    "DEFENSE_PROFILED",
    "DATASET_MANIFEST_CREATED",
    "TRAINING_STARTED",
    "TRAINING_COMPLETED",
    "CHECKPOINT_VALIDATED",
    "GATE_EVALUATED",
    "MODEL_REGISTERED",
    "RE_BENCHMARK_STARTED",
    "RE_BENCHMARK_COMPLETED",
    "RECOVERY_REPORTED",
    "FAILED",
    "CANCELLED",
]

TERMINAL_STATES = frozenset({"RECOVERY_REPORTED", "FAILED", "CANCELLED"})

# Valid transitions — ordered sequence with failure/cancel from any state
_NEXT_STATES: dict[str, set[str]] = {
    "FAILURE_IDENTIFIED": {"CLUSTER_FORMED", "FAILED"},
    "CLUSTER_FORMED": {"BACKLOG_CREATED", "FAILED"},
    "BACKLOG_CREATED": {"BACKLOG_APPROVED", "FAILED", "CANCELLED"},
    "BACKLOG_APPROVED": {"DEFENSE_PROFILED", "FAILED", "CANCELLED"},
    "DEFENSE_PROFILED": {"DATASET_MANIFEST_CREATED", "FAILED", "CANCELLED"},
    "DATASET_MANIFEST_CREATED": {"TRAINING_STARTED", "FAILED", "CANCELLED"},
    "TRAINING_STARTED": {"TRAINING_COMPLETED", "FAILED", "CANCELLED"},
    "TRAINING_COMPLETED": {"CHECKPOINT_VALIDATED", "FAILED"},
    "CHECKPOINT_VALIDATED": {"GATE_EVALUATED", "FAILED"},
    "GATE_EVALUATED": {"MODEL_REGISTERED", "FAILED"},
    "MODEL_REGISTERED": {"RE_BENCHMARK_STARTED", "FAILED"},
    "RE_BENCHMARK_STARTED": {"RE_BENCHMARK_COMPLETED", "FAILED", "CANCELLED"},
    "RE_BENCHMARK_COMPLETED": {"RECOVERY_REPORTED", "FAILED"},
}


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class TrainingDatasetManifest(_FrozenContract):
    """Full lineage of the dataset used for a training run."""

    manifest_id: str
    source_dataset_version_id: str
    generated_dataset_ids: tuple[str, ...] = ()
    sample_ids: tuple[str, ...] = ()
    source_hashes: tuple[str, ...] = ()
    clean_sample_count: int = Field(ge=0)
    generated_sample_count: int = Field(ge=0)
    clean_ratio: float = Field(ge=0.0, le=1.0)
    defense_profile_id: str
    recipe_ids: tuple[str, ...] = ()
    seed: int = Field(ge=0)
    locked_test_excluded: bool = True
    leakage_report_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClosedLoopAuditEntry(_FrozenContract):
    """One step in the closed-loop retraining audit trail."""

    step: int = Field(ge=0)
    state: ClosedLoopState
    artifact_id: str
    artifact_type: str
    artifact_hash: str | None = None
    parent_step: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecoveryReport(_FrozenContract):
    """Paired comparison showing what the retraining fixed."""

    report_id: str
    baseline_model_id: str
    candidate_model_id: str
    protocol_id: str
    baseline_run_id: str
    candidate_run_id: str
    paired: bool = True
    recovery_rate: float | None = None
    clean_delta: float = 0.0
    robust_score_delta: float = 0.0
    degradation_delta: float = 0.0
    failure_count_delta: int = 0
    gate_passed: bool = False
    audit_trail: tuple[ClosedLoopAuditEntry, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_transition(current: ClosedLoopState, target: ClosedLoopState) -> bool:
    """Check if a state transition is valid."""
    if current in TERMINAL_STATES:
        return False
    return target in _NEXT_STATES.get(current, set())


@dataclass
class ClosedLoopTracker:
    """Tracks progress through the closed-loop retraining pipeline."""

    loop_id: str
    state: ClosedLoopState = "FAILURE_IDENTIFIED"
    audit: list[ClosedLoopAuditEntry] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def advance(
        self,
        target: ClosedLoopState,
        artifact_id: str,
        artifact_type: str,
        *,
        artifact_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClosedLoopAuditEntry:
        """Advance to next state if transition is valid."""
        if not validate_transition(self.state, target):
            raise ValueError(
                f"Invalid transition: {self.state} → {target}. Valid: {sorted(_NEXT_STATES.get(self.state, set()))}"
            )
        entry = ClosedLoopAuditEntry(
            step=len(self.audit),
            state=target,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            artifact_hash=artifact_hash,
            parent_step=len(self.audit) - 1 if self.audit else None,
            metadata=metadata or {},
        )
        self.audit.append(entry)
        self.artifacts[artifact_type] = artifact_id
        self.state = target
        return entry

    def fail(self, reason: str) -> ClosedLoopAuditEntry:
        """Transition to FAILED state from any non-terminal state."""
        return self.advance(
            "FAILED",
            artifact_id=f"failure-{stable_digest({'reason': reason}, length=12)}",
            artifact_type="failure_record",
            metadata={"reason": reason},
        )

    @property
    def is_complete(self) -> bool:
        return self.state in TERMINAL_STATES

    def lineage_chain(self) -> list[str]:
        """Return ordered list of artifact IDs forming the lineage."""
        return [entry.artifact_id for entry in self.audit]
