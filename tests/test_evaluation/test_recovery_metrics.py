import pytest

from src.evaluation.recovery_metrics import UndefinedMetric, recovery_rate


def test_recovery_rate_handles_direction_and_unbounded_results() -> None:
    assert recovery_rate(1.0, 0.5, 0.8).value == pytest.approx(0.6)
    assert recovery_rate(1.0, 0.5, 1.1).value == pytest.approx(1.2)
    assert recovery_rate(1.0, 0.5, 0.4).value == pytest.approx(-0.2)
    assert recovery_rate(0.1, 0.5, 0.3, higher_is_better=False).value == pytest.approx(0.5)


def test_recovery_rate_marks_zero_denominator_undefined() -> None:
    result = recovery_rate(0.5, 0.5, 0.6)
    assert isinstance(result, UndefinedMetric)
    assert "denominator" in result.reason
