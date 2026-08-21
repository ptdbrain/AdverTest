"""Analytics aggregation package for single runs and paired model comparisons."""

from src.analytics.comparison_analytics import (
    compute_comparison_classes,
    compute_comparison_failures,
    compute_comparison_recovery,
    compute_comparison_summary,
)
from src.analytics.run_analytics import (
    compute_run_attacks_breakdown,
    compute_run_classes_breakdown,
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
    "compute_comparison_summary",
    "compute_comparison_recovery",
    "compute_comparison_classes",
    "compute_comparison_failures",
]
