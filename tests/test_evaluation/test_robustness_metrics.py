"""Aggregate robustness metrics (plan §3 metrics 3, 4, 5, 7, 13)."""

from __future__ import annotations

import pytest

from src.evaluation.report import CellResult, RunReport
from src.evaluation.robustness_metrics import (
    ap_grid,
    bootstrap_ci,
    category_scores,
    covered_categories,
    degradation_metrics,
    mpc,
    resilience_rate,
    robust_score,
    robustness_accuracy,
    rpc,
    severity_monotonicity,
    summary,
)


def _report(*cells: CellResult, ap_clean: float = 1.0) -> RunReport:
    return RunReport(
        run_id="test",
        model="m",
        model_version="m-1",
        dataset="d",
        n_samples=10,
        ap_clean=ap_clean,
        cells=list(cells),
    )


def _cell(attack: str, group: str, severity: int, ap: float) -> CellResult:
    return CellResult(attack=attack, group=group, severity=severity, ap=ap, n_samples=10)  # type: ignore[arg-type]


#: Two group C attacks, two severities each; clean AP is 1.0 so ratios are readable.
SIMPLE = _report(
    _cell("random_erasing", "C", 1, 0.8),
    _cell("random_erasing", "C", 5, 0.4),
    _cell("object_occlusion", "C", 1, 0.6),
    _cell("object_occlusion", "C", 5, 0.2),
)


def test_ap_grid_indexes_by_attack_then_severity() -> None:
    assert ap_grid(SIMPLE)["random_erasing"] == {1: 0.8, 5: 0.4}


def test_mpc_averages_over_severities_then_attacks() -> None:
    # (0.8 + 0.4)/2 = 0.6 ; (0.6 + 0.2)/2 = 0.4 ; mean = 0.5
    assert mpc(SIMPLE) == pytest.approx(0.5)


def test_rpc_is_mpc_relative_to_clean_ap() -> None:
    assert rpc(_report(*SIMPLE.cells, ap_clean=0.5)) == pytest.approx(1.0)


def test_resilience_rate_is_per_attack() -> None:
    assert resilience_rate(SIMPLE) == pytest.approx({"random_erasing": 0.6, "object_occlusion": 0.4})


def test_robustness_accuracy_is_the_severity_curve() -> None:
    assert robustness_accuracy(SIMPLE) == pytest.approx({1: 0.7, 5: 0.3})


def test_zero_clean_ap_does_not_divide_by_zero() -> None:
    empty = _report(_cell("x", "C", 1, 0.0), ap_clean=0.0)
    assert rpc(empty) == 0.0
    assert robust_score(empty) == 0.0


# ------------------------------------------------------------- RobustScore


def test_category_scores_group_by_the_plan_categories() -> None:
    mixed = _report(_cell("gaussian_noise", "A", 1, 0.5), _cell("random_erasing", "C", 1, 0.9))
    assert category_scores(mixed) == pytest.approx({"noise": 0.5, "occlusion": 0.9})


def test_plan_formula_caps_a_single_category_run_at_25() -> None:
    """Faithful to plan §3 metric 13: unmeasured categories contribute nothing."""
    perfect = _report(_cell("random_erasing", "C", 1, 1.0))
    assert robust_score(perfect) == pytest.approx(25.0)
    assert covered_categories(perfect) == ["occlusion"]


def test_normalized_score_rescales_to_what_was_measured() -> None:
    perfect = _report(_cell("random_erasing", "C", 1, 1.0))
    assert robust_score(perfect, normalize=True) == pytest.approx(100.0)


def test_weights_are_configurable() -> None:
    report = _report(_cell("random_erasing", "C", 1, 0.5))
    assert robust_score(report, {"occlusion": 1.0}) == pytest.approx(50.0)


# ----------------------------------------------------------- sanity check #2


def test_monotonicity_flags_a_rising_ladder() -> None:
    good = _report(_cell("a", "C", 1, 0.8), _cell("a", "C", 2, 0.5))
    bad = _report(_cell("b", "C", 1, 0.5), _cell("b", "C", 2, 0.8))
    assert severity_monotonicity(good) == {"a": True}
    assert severity_monotonicity(bad) == {"b": False}


def test_bootstrap_ci_returns_percentile_bounds() -> None:
    low, high = bootstrap_ci([float(value) / 100.0 for value in range(101)])
    assert low == pytest.approx(0.025, abs=0.01)
    assert high == pytest.approx(0.975, abs=0.01)


def test_bootstrap_ci_handles_no_replicates() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_summary_carries_every_headline_number() -> None:
    payload = summary(SIMPLE)
    for key in ("ap_clean", "mpc", "rpc", "robust_score_plan", "robust_score_normalized"):
        assert key in payload
    assert payload["covered_categories"] == ["occlusion"]


def test_degradation_metrics_have_explicit_units_and_direction() -> None:
    metrics = degradation_metrics(0.8, 0.6, higher_is_better=True)
    assert metrics["degradation_ratio"].unit == "ratio"
    assert metrics["degradation_ratio"].percent_value == pytest.approx(25.0)
    assert metrics["absolute_point_delta"].value == pytest.approx(-0.2)
    assert metrics["relative_change"].higher_is_better is True
