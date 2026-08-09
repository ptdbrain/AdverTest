"""Request/response DTOs for the HTTP layer.

Domain objects (``Sample``, ``Box``, ``RunReport``) stay in the core; these
models exist so the API contract can change without touching the pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.attacks.recipes import AttackRecipe
from src.core.objectives import RequiredAnnotation, SurrogateCapability
from src.core.types import Modality, Task
from src.models.versions import ModelVersion


class AttackCatalogItem(BaseModel):
    """One entry of the attack catalog (mirrors ``BaseAttack.describe``)."""

    name: str
    version: str = "1.0.0"
    group: str
    title: str = ""
    modality: str
    cost_class: str
    severity_levels: int
    needs_model: bool
    needs_gradients: bool
    required_annotations: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    required_tasks: list[str] = Field(default_factory=list)
    required_sensors: list[str] = Field(default_factory=list)
    affected_sensors: list[str] = Field(default_factory=list)
    generation_mode: str = "per_sample"
    category: str = "adversarial"
    owner: str
    reference: str = ""
    params_schema: dict[str, Any] = Field(default_factory=dict)
    catalog_version: str = "1.0.0"
    implementation_version: str = "1.0.0"
    display_name: str = ""
    plain_summary: str = ""
    technical_summary: str = ""
    scenario: str = ""
    rationale: str = ""
    failure_symptoms: list[str] = Field(default_factory=list)
    severity_map: dict[int, str] = Field(default_factory=dict)
    compatibility: dict[str, Any] = Field(default_factory=dict)
    runtime_class: str = "instant"
    defense_hint: str = ""
    deterministic: bool = True
    supports_online: bool = True
    supports_offline: bool = True
    production_status: str = "production"


class ModelCatalogItem(BaseModel):
    """One entry of the model-adapter catalog."""

    name: str
    task: str
    version: str
    modality: str
    supports_gradients: bool
    capabilities: list[str] = Field(default_factory=list)
    runnable: bool = True
    owner: str
    docstring: str = ""


class ModelVersionOut(BaseModel):
    """A locally registered checkpoint without exposing its bytes."""

    model_config = ConfigDict(extra="forbid")

    id: str
    model_name: str
    task: str
    checkpoint_path: str | None = None
    checkpoint_hash: str | None = None
    parent_id: str | None = None
    training_metadata: dict[str, Any] = Field(default_factory=dict)
    runnable: bool
    blocked_reason: str | None = None

    @classmethod
    def from_domain(cls, version: ModelVersion) -> ModelVersionOut:
        return cls(
            id=version.id,
            model_name=version.model_name,
            task=version.task,
            checkpoint_path=version.checkpoint_path,
            checkpoint_hash=version.checkpoint_hash,
            parent_id=version.parent_id,
            training_metadata=dict(version.training_metadata),
            runnable=version.runnable,
            blocked_reason=version.blocked_reason,
        )


class PerceptionModeOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Task
    title: str
    metric_labels: tuple[str, ...]
    runnable: bool
    blocked_reason: str | None = None


class RecipeValidationIn(BaseModel):
    """Context required to validate a recipe before a long-running job."""

    model_config = ConfigDict(extra="forbid")

    recipe: AttackRecipe
    task: Task
    model_capabilities: frozenset[SurrogateCapability] = frozenset()
    annotation_types: frozenset[RequiredAnnotation] = frozenset()
    modality: Modality = "image"
    online: bool = False
    requested_variants: int = Field(default=1, ge=1)
    bytes_per_variant: int = Field(default=0, ge=0)


class RecipeValidationOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    estimate: dict[str, float | int]


class RecipeRecordIn(BaseModel):
    """Persisted, user-owned recipe metadata; steps remain explicit JSON."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    name: str = Field(min_length=1, max_length=200)
    seed: int = Field(ge=0)
    steps: list[dict[str, Any]] = Field(default_factory=list)


class DatasetImportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    logical_source_id: str = Field(min_length=1, max_length=200)
    input_format: str = Field(default="advertest", pattern="^(advertest|kitti)$")
    anonymization_manifest: str | None = None
    max_samples: int | None = Field(default=None, ge=1)


class ModelComparisonIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str = Field(min_length=1)
    candidate_run_id: str = Field(min_length=1)


class DatasetCatalogItem(BaseModel):
    """One entry of the dataset catalog; ``anonymized`` gates test runs."""

    name: str
    title: str = ""
    anonymized: bool
    modality: str
    owner: str
    params_schema: dict[str, Any] = Field(default_factory=dict)


class CostEstimateOut(BaseModel):
    """Pre-run estimate — plan §5 requires this before a run may start."""

    n_cells: int
    n_samples: int
    n_forward_passes: int
    n_model_queries: int = 0
    n_gradient_steps: int = 0
    cost_units: float
    estimated_seconds: float


class CellOut(BaseModel):
    """One ``(attack, severity)`` cell of the report grid."""

    attack: str
    group: str
    severity: int
    ap: float
    degradation: float
    n_samples: int
    seconds: float
    cache_hits: int
    category: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class SkippedOut(BaseModel):
    attack: str
    reason: str


class RunReportOut(BaseModel):
    """Full report for one test run."""

    run_id: str
    model: str
    model_version: str
    dataset: str
    n_samples: int
    ap_clean: float
    cells: list[CellOut] = Field(default_factory=list)
    heatmap: dict[str, dict[int, float]] = Field(default_factory=dict)
    worst_cases: list[dict[str, Any]] = Field(default_factory=list)
    skipped: list[SkippedOut] = Field(default_factory=list)
    sample_results: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    seconds: float = 0.0
    #: Constant reminder that nothing here validates a model for deployment.
    simulation_only: bool = True


class RunSummaryOut(BaseModel):
    """Compact row for the run list."""

    run_id: str
    model: str
    dataset: str
    ap_clean: float
    n_cells: int
    worst_degradation: float
    needs_review: bool


class PreflightOut(BaseModel):
    compatible: list[str] = Field(default_factory=list)
    skipped_with_reason: list[SkippedOut] = Field(default_factory=list)
    fatal_errors: list[str] = Field(default_factory=list)


class RunJobOut(BaseModel):
    run_id: str
    status: str
    progress: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)
    report: RunReportOut | None = None
    error: str | None = None


class ReviewOut(BaseModel):
    """Review queue item."""

    review_id: str
    run_id: str
    attack: str
    severity: int
    dataset: str = ""
    model: str = ""
    degradation: float = 0.0
    status: str = "PENDING"
    decision: str | None = None
    decision_note: str | None = None
    flagged_by: str = "system_auto"
    resolved_by: str | None = None
    created_at: str = ""
    updated_at: str = ""


class CreateReviewIn(BaseModel):
    """Manual review creation request."""

    run_id: str
    attack: str
    severity: int
    degradation: float = 0.0
    dataset: str = ""
    model: str = ""
    flagged_by: str = "manual"
    notes: str = ""


class ResolveReviewIn(BaseModel):
    """Review resolution request."""

    decision: str  # ACCEPT_RISK | REQUEST_RETRAIN
    decision_note: str
    resolved_by: str = "reviewer"
