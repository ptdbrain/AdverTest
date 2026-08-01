"""Request/response DTOs for the HTTP layer.

Domain objects (``Sample``, ``Box``, ``RunReport``) stay in the core; these
models exist so the API contract can change without touching the pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
    generation_mode: str = "per_sample"
    owner: str
    reference: str = ""
    params_schema: dict[str, Any] = Field(default_factory=dict)


class ModelCatalogItem(BaseModel):
    """One entry of the model-adapter catalog."""

    name: str
    task: str
    version: str
    modality: str
    supports_gradients: bool
    capabilities: list[str] = Field(default_factory=list)
    owner: str
    docstring: str = ""


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
