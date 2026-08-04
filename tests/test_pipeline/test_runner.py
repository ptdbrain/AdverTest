"""End-to-end pipeline behaviour: cells, caching, reproducibility, skip reasons."""

from __future__ import annotations

import pytest

from src.attacks import load_attacks
from src.core.types import ModelInfo
from src.datasets import get_dataset
from src.pipeline import MemoryCache, NullCache, RunConfig, TestRunner
from src.pipeline.runner import _incompatibility

CONFIG = RunConfig(attacks=["gaussian_noise"], severities=[1, 5], limit=3, bootstrap_repetitions=0)


def test_run_produces_one_cell_per_attack_and_severity() -> None:
    report = TestRunner().run(CONFIG)
    assert len(report.cells) == 2
    assert {cell.severity for cell in report.cells} == {1, 5}
    assert report.n_samples == 3
    assert report.simulation_only is True


def test_report_bootstrap_resamples_samples_not_aggregate_cells() -> None:
    report = TestRunner().run(
        RunConfig(attacks=["gaussian_noise"], severities=[1], limit=2, bootstrap_repetitions=20)
    )
    assert report.metrics["clean"]["ap50_ci95"] is not None
    assert report.cells[0].metrics["ap50_ci95"] is not None


def test_clean_baseline_is_strong_enough_to_measure_against() -> None:
    report = TestRunner().run(CONFIG)
    assert report.ap_clean >= 0.8, "a weak baseline makes every degradation number meaningless"


def test_stronger_severity_degrades_at_least_as_much() -> None:
    report = TestRunner().run(CONFIG)
    by_severity = {cell.severity: report.degradation(cell) for cell in report.cells}
    assert by_severity[5] >= by_severity[1]


def test_second_identical_run_is_served_from_cache() -> None:
    runner = TestRunner(MemoryCache())
    runner.run(CONFIG)
    second = runner.run(CONFIG)
    assert sum(cell.cache_hits for cell in second.cells) == 2 * CONFIG.limit


def test_runs_are_reproducible_without_a_cache() -> None:
    first = TestRunner(NullCache()).run(CONFIG)
    second = TestRunner(NullCache()).run(CONFIG)
    assert [cell.ap for cell in first.cells] == [cell.ap for cell in second.cells]


def test_empty_attack_list_means_the_whole_catalog() -> None:
    report = TestRunner().run(RunConfig(severities=[1], limit=2))
    covered = {cell.attack for cell in report.cells} | {item.attack for item in report.skipped}
    assert covered == set(load_attacks().names())


def test_estimate_matches_the_executed_run() -> None:
    runner = TestRunner()
    estimate = runner.estimate(CONFIG)
    report = runner.run(CONFIG)
    assert estimate.n_cells == len(report.cells)
    assert estimate.n_forward_passes == CONFIG.limit * (len(report.cells) + 1)
    assert estimate.estimated_seconds > 0.0


def test_square_cost_counts_its_query_budget() -> None:
    estimate = TestRunner().estimate(RunConfig(attacks=["square_attack"], severities=[1], limit=2))
    assert estimate.n_model_queries == 1000
    assert estimate.n_forward_passes >= 1000


def test_preflight_skips_capability_and_severity_mismatches() -> None:
    result = TestRunner().preflight(RunConfig(attacks=["sam2_pgd", "square_attack"], severities=[5], limit=1))
    reasons = {item.attack: item.reason for item in result.skipped}
    assert "sam2_pgd" in reasons and "model task" in reasons["sam2_pgd"]
    assert "square_attack" in reasons and "severities" in reasons["square_attack"]


def test_unknown_attack_is_reported_not_ignored() -> None:
    with pytest.raises(Exception, match="unknown attack"):
        TestRunner().run(RunConfig(attacks=["nope_not_here"], limit=1))


def test_gradient_attack_is_skipped_on_a_black_box_model() -> None:
    """Skips must always carry a reason (plan §11: no silent gaps in a report)."""
    black_box = ModelInfo(name="black_box", task="detection2d", version="1", supports_gradients=False)
    dataset = get_dataset("synthetic_shapes", n_samples=1)
    reason = _incompatibility(load_attacks().get("fgsm"), dataset, black_box)
    assert reason is not None and "gradient" in reason


def test_image_attack_runs_on_an_image_dataset() -> None:
    info = ModelInfo(name="m", task="detection2d", version="1", supports_gradients=True)
    dataset = get_dataset("synthetic_shapes", n_samples=1)
    assert _incompatibility(load_attacks().get("gaussian_noise"), dataset, info) is None


def test_square_attack_requires_a_2d_detector() -> None:
    info = ModelInfo(name="segmenter", task="segmentation", version="1")
    dataset = get_dataset("synthetic_shapes", n_samples=1)
    reason = _incompatibility(load_attacks().get("square_attack"), dataset, info)
    assert reason is not None and "model task" in reason
