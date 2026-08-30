from typing import Any

from pydantic import BaseModel, Field


class CostEstimateOut(BaseModel):
    n_cells: int
    n_samples: int
    n_forward_passes: int
    n_model_queries: int = 0
    n_gradient_steps: int = 0
    cost_units: float
    estimated_seconds: float
    estimate_token: str | None = None
    artifact_storage_estimate_bytes: int = 0
    gpu_cpu_cost_estimate: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class SkippedOut(BaseModel):
    attack: str
    reason: str


class CellOut(BaseModel):
    attack: str
    group: str
    severity: int
    ap: float
    degradation: float
    degradation_ratio: float = 0.0
    degradation_percent: float = 0.0
    unit: str = "ratio"
    n_samples: int
    seconds: float
    cache_hits: int
    category: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class RunReportOut(BaseModel):
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
    simulation_only: bool = True
    benchmark_metrics_available: bool = False
    needs_review: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)
    promotion_eligible: bool = False


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
