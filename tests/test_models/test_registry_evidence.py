"""Regression coverage for the model catalogue's evidence boundary."""

from src.models.registry import list_model_versions


def test_catalogue_never_publishes_unverified_benchmark_kpis() -> None:
    """Changing a model card must not silently turn illustrative scores into product KPIs."""
    assert all(version.metrics == {} for version in list_model_versions())
