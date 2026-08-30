"""Wave 0 regression tests: metric unit standardisation and degradation display.

Covers: 0, 0.0042, 0.42, 1.0, missing metric, undefined metric, and edge cases.
Acceptance: `0.42 → 42.0%`; backend does not double-multiply or forget to multiply.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.types import Box, Sample
from src.evaluation.contracts import MetricEnvelope
from src.evaluation.report import CellResult, RunReport
from src.evaluation.robustness_metrics import degradation_metrics

# ---- Helpers ----


def _sample(sample_id: str = "s0") -> Sample:
    return Sample(
        sample_id=sample_id,
        image=np.zeros((64, 64, 3), dtype=np.float32),
        boxes=(Box(10, 10, 50, 50, "Car"),),
    )


def _report(ap_clean: float, cells: list[CellResult]) -> RunReport:
    return RunReport(
        run_id="test",
        model="blob_detector",
        model_version="v0",
        dataset="synthetic",
        n_samples=4,
        ap_clean=ap_clean,
        cells=cells,
    )


def _cell(attack: str, severity: int, ap: float) -> CellResult:
    return CellResult(
        attack=attack,
        group="A",
        severity=severity,
        ap=ap,
        n_samples=4,
    )


# ---- Unit conversion tests ----


class TestDegradationUnitConversion:
    """Verify `0.42 → 42.0%` consistently across domain, API, and export."""

    def test_ratio_0_means_no_degradation(self):
        report = _report(0.80, [_cell("noise", 1, 0.80)])
        serialized = report.as_dict()
        cell = serialized["cells"][0]
        assert cell["degradation_ratio"] == 0.0
        assert cell["degradation_percent"] == 0.0

    def test_ratio_042_becomes_42_percent(self):
        """The defining acceptance criterion: 0.42 ratio → 42.0% display."""
        report = _report(1.0, [_cell("noise", 1, 0.58)])
        serialized = report.as_dict()
        cell = serialized["cells"][0]
        assert math.isclose(cell["degradation_ratio"], 0.42, abs_tol=1e-4)
        assert math.isclose(cell["degradation_percent"], 42.0, abs_tol=0.1)

    def test_ratio_00042_becomes_042_percent(self):
        report = _report(1.0, [_cell("noise", 1, 0.9958)])
        serialized = report.as_dict()
        cell = serialized["cells"][0]
        assert math.isclose(cell["degradation_ratio"], 0.0042, abs_tol=1e-4)
        assert math.isclose(cell["degradation_percent"], 0.42, abs_tol=0.1)

    def test_ratio_1_means_total_failure(self):
        report = _report(0.80, [_cell("noise", 5, 0.0)])
        serialized = report.as_dict()
        cell = serialized["cells"][0]
        assert math.isclose(cell["degradation_ratio"], 1.0, abs_tol=1e-4)
        assert math.isclose(cell["degradation_percent"], 100.0, abs_tol=0.1)

    def test_zero_clean_ap_means_zero_degradation(self):
        """Edge: clean AP is 0 → degradation is defined as 0, not inf/NaN."""
        report = _report(0.0, [_cell("noise", 1, 0.0)])
        serialized = report.as_dict()
        cell = serialized["cells"][0]
        assert cell["degradation_ratio"] == 0.0
        assert cell["degradation_percent"] == 0.0
        assert math.isfinite(cell["degradation_ratio"])

    def test_every_cell_has_unit_field(self):
        report = _report(
            0.80,
            [
                _cell("noise", 1, 0.60),
                _cell("fog", 2, 0.40),
            ],
        )
        for cell in report.as_dict()["cells"]:
            assert "unit" in cell
            assert cell["unit"] == "ratio"

    def test_legacy_degradation_preserved(self):
        """The deprecated field must still exist for backward compat."""
        report = _report(1.0, [_cell("noise", 3, 0.58)])
        cell = report.as_dict()["cells"][0]
        assert "degradation" in cell
        assert math.isclose(cell["degradation"], cell["degradation_ratio"], abs_tol=1e-4)


class TestMetricEnvelopeContract:
    """Verify MetricEnvelope enforces ratio ↔ percent consistency."""

    def test_ratio_metric_requires_percent_value(self):
        with pytest.raises(ValueError, match="percent_value"):
            MetricEnvelope(
                name="test",
                value=0.42,
                unit="ratio",
                percent_value=None,
                version="1.0.0",
                higher_is_better=True,
            )

    def test_ratio_metric_percent_must_match(self):
        with pytest.raises(ValueError, match="percent_value must equal"):
            MetricEnvelope(
                name="test",
                value=0.42,
                unit="ratio",
                percent_value=43.0,  # wrong: should be 42.0
                version="1.0.0",
                higher_is_better=True,
            )

    def test_valid_ratio_metric(self):
        m = MetricEnvelope(
            name="degradation",
            value=0.42,
            unit="ratio",
            percent_value=42.0,
            version="1.0.0",
            higher_is_better=False,
        )
        assert m.value == 0.42
        assert m.percent_value == 42.0

    def test_percent_metric_consistency(self):
        m = MetricEnvelope(
            name="degradation_pct",
            value=42.0,
            unit="percent",
            percent_value=42.0,
            version="1.0.0",
            higher_is_better=False,
        )
        assert m.value == 42.0
        assert m.percent_value == 42.0

    def test_points_metric_no_percent_needed(self):
        m = MetricEnvelope(
            name="ap_delta",
            value=-2.5,
            unit="points",
            percent_value=None,
            version="1.0.0",
            higher_is_better=True,
        )
        assert m.percent_value is None

    def test_count_metric(self):
        m = MetricEnvelope(
            name="failure_count",
            value=17.0,
            unit="count",
            percent_value=None,
            version="1.0.0",
            higher_is_better=False,
        )
        assert m.value == 17.0

    def test_empty_version_rejected(self):
        with pytest.raises(ValueError, match="version"):
            MetricEnvelope(
                name="test",
                value=0.5,
                unit="ratio",
                percent_value=50.0,
                version="   ",
                higher_is_better=True,
            )


class TestDegradationMetrics:
    """Test the degradation_metrics helper produces all four unit-explicit metrics."""

    def test_degradation_042(self):
        result = degradation_metrics(1.0, 0.58)
        ratio = result["degradation_ratio"]
        pct = result["degradation_pct"]
        assert math.isclose(ratio.value, 0.42, abs_tol=1e-4)
        assert ratio.unit == "ratio"
        assert math.isclose(ratio.percent_value, 42.0, abs_tol=0.1)
        assert math.isclose(pct.value, 42.0, abs_tol=0.1)
        assert pct.unit == "percent"

    def test_zero_degradation(self):
        result = degradation_metrics(0.8, 0.8)
        assert result["degradation_ratio"].value == 0.0
        assert result["degradation_pct"].value == 0.0

    def test_total_failure(self):
        result = degradation_metrics(0.8, 0.0)
        assert math.isclose(result["degradation_ratio"].value, 1.0, abs_tol=1e-4)
        assert math.isclose(result["degradation_pct"].value, 100.0, abs_tol=0.1)

    def test_zero_clean_no_error(self):
        result = degradation_metrics(0.0, 0.0)
        assert result["degradation_ratio"].value == 0.0


class TestHeatmapUnitConsistency:
    """Heatmap must produce ratio values in [0, 1], not percent."""

    def test_heatmap_values_are_ratios(self):
        report = _report(
            1.0,
            [
                _cell("noise", 1, 0.90),
                _cell("noise", 3, 0.58),
                _cell("fog", 2, 0.50),
            ],
        )
        heatmap = report.heatmap()
        for attack, severities in heatmap.items():
            for severity, value in severities.items():
                assert 0.0 <= value <= 1.0, f"heatmap[{attack}][{severity}]={value} not in [0,1]"

    def test_heatmap_042(self):
        report = _report(1.0, [_cell("noise", 1, 0.58)])
        heatmap = report.heatmap()
        assert math.isclose(heatmap["noise"][1], 0.42, abs_tol=1e-4)
