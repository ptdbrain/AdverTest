import pytest

from src.evaluation.recovery_metrics import recovery_rate
from src.services.person_d import PersonDServices


def test_detection_platform_vertical_slice_contracts_are_wired() -> None:
    services = PersonDServices.default()
    assert services.benchmarks.runner_type.__name__ == "BenchmarkRunner"
    assert services.training_data.builder.__class__.__name__ == "TrainingDatasetBuilder"
    assert recovery_rate(0.9, 0.5, 0.7).value == pytest.approx(0.5)  # type: ignore[union-attr]
