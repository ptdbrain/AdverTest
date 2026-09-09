"""Analytics aggregation package for single runs, project trends, and paired model comparisons."""

from src.analytics.comparison_analytics import (
    compute_comparison_classes,
    compute_comparison_failures,
    compute_comparison_recovery,
    compute_comparison_summary,
)
from src.analytics.project_analytics import compute_project_analytics, compute_runs_comparison
from src.analytics.run_analytics import (
    compute_run_attacks_breakdown,
    compute_run_classes_breakdown,
    compute_run_distance_breakdown,
    compute_run_samples_breakdown,
    compute_run_severity_breakdown,
    compute_run_summary,
)

__all__ = [
    "compute_run_summary",
    "compute_run_attacks_breakdown",
    "compute_run_severity_breakdown",
    "compute_run_classes_breakdown",
    "compute_run_samples_breakdown",
    "compute_run_distance_breakdown",
    "compute_project_analytics",
    "compute_runs_comparison",
    "compute_comparison_summary",
    "compute_comparison_recovery",
    "compute_comparison_classes",
    "compute_comparison_failures",
]
