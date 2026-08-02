"""Aggregate robustness metrics of plan §3, computed from a finished RunReport.

These are the numbers a benchmark reports on top of the per-cell AP that
:class:`~src.evaluation.report.RunReport` already carries:

===============  ==========================================================
Plan §3 metric   Function
===============  ==========================================================
3  ``mPC``       :func:`mpc`
4  ``rPC``       :func:`rpc`
5  ``RR(c)``     :func:`resilience_rate`
7  ``RA(s)``     :func:`robustness_accuracy`
13 RobustScore   :func:`robust_score`
===============  ==========================================================

Everything is a pure function of the report, so nothing here needs to be wired
into the runner and no shared file changes.

A caveat worth reading before quoting a RobustScore: plan §3 defines it over four
categories (weather, noise, occlusion, adversarial) each weighted 0.25, so a run
that only measured one category cannot score above 25 out of 100. That is the
formula as written, and :func:`robust_score` reproduces it. Pass
``normalize=True`` to divide by the weight actually covered — useful while only
part of the catalog exists, dishonest if reported as "the" RobustScore without
saying so.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.types import GROUP_CATEGORY
from src.evaluation.report import RunReport

#: Plan §3 metric 13: equal weight per category unless the caller says otherwise.
DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    "weather": 0.25,
    "noise": 0.25,
    "occlusion": 0.25,
    "adversarial": 0.25,
}


def ap_grid(report: RunReport) -> dict[str, dict[int, float]]:
    """``{attack: {severity: AP(c, s)}}`` — the raw grid behind every aggregate."""
    grid: dict[str, dict[int, float]] = {}
    for cell in report.cells:
        grid.setdefault(cell.attack, {})[cell.severity] = cell.ap
    return grid


def severities(report: RunReport) -> list[int]:
    """Severity levels present in the report, ascending."""
    return sorted({cell.severity for cell in report.cells})


def mpc(report: RunReport) -> float:
    """Plan §3 metric 3 — mean over attacks of the mean over severities of AP."""
    per_attack = [float(np.mean(list(row.values()))) for row in ap_grid(report).values() if row]
    return float(np.mean(per_attack)) if per_attack else 0.0


def rpc(report: RunReport) -> float:
    """Plan §3 metric 4 — fraction of clean AP retained under corruption."""
    return _ratio(mpc(report), report.ap_clean)


def resilience_rate(report: RunReport) -> dict[str, float]:
    """Plan §3 metric 5 — ``RR(c) = mean_s AP(c, s) / AP_clean`` per attack."""
    return {
        attack: _ratio(float(np.mean(list(row.values()))), report.ap_clean)
        for attack, row in ap_grid(report).items()
        if row
    }


def robustness_accuracy(report: RunReport) -> dict[int, float]:
    """Plan §3 metric 7 — ``RA(s) = mean_c AP(c, s) / AP_clean``, the severity curve."""
    by_severity: dict[int, list[float]] = {}
    for cell in report.cells:
        by_severity.setdefault(cell.severity, []).append(cell.ap)
    return {
        severity: _ratio(float(np.mean(values)), report.ap_clean)
        for severity, values in sorted(by_severity.items())
    }


def category_scores(report: RunReport) -> dict[str, float]:
    """Retained AP fraction per RobustScore category (weather / noise / …)."""
    by_category: dict[str, list[float]] = {}
    for cell in report.cells:
        by_category.setdefault(GROUP_CATEGORY[cell.group], []).append(cell.ap)
    return {
        category: float(np.clip(_ratio(float(np.mean(values)), report.ap_clean), 0.0, 1.0))
        for category, values in sorted(by_category.items())
    }


def robust_score(
    report: RunReport,
    weights: dict[str, float] | None = None,
    *,
    normalize: bool = False,
) -> float:
    """Plan §3 metric 13 — one 0-100 number per model version.

    ``normalize=True`` rescales by the weight of the categories that were
    actually measured, so a group-C-only run is not capped at 25.
    """
    applied = weights if weights is not None else DEFAULT_CATEGORY_WEIGHTS
    scores = category_scores(report)
    total = sum(applied.get(category, 0.0) * score for category, score in scores.items())
    if normalize:
        covered = sum(applied.get(category, 0.0) for category in scores)
        return float(100.0 * _ratio(total, covered))
    return float(100.0 * total)


def covered_categories(report: RunReport) -> list[str]:
    """Categories the run touched — context for reading a RobustScore."""
    return sorted({GROUP_CATEGORY[cell.group] for cell in report.cells})


def severity_monotonicity(report: RunReport) -> dict[str, bool]:
    """Sanity check #2 of plan §3: AP must not rise as severity rises.

    A violation means either the severity ladder is mis-specified or the sample
    is too small for the AP estimate to be stable — both are reasons not to
    publish the row, which is why the benchmark prints this as a pass/fail table.
    """
    checks: dict[str, bool] = {}
    for attack, row in ap_grid(report).items():
        ordered = [row[severity] for severity in sorted(row)]
        checks[attack] = all(
            later <= earlier + 1e-9 for earlier, later in zip(ordered, ordered[1:], strict=False)
        )
    return checks


def bootstrap_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval of an already-computed bootstrap distribution.

    Plan §3 requires every reported AP to carry a 95 % interval and forbids
    comparing two models whose intervals overlap. Producing the replicates is the
    caller's job (resample sample ids, recompute AP from cached predictions).
    """
    if not values:
        return (0.0, 0.0)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(np.asarray(values, dtype=np.float64), [tail, 1.0 - tail])
    return (float(low), float(high))


def summary(report: RunReport, *, normalize_score: bool = True) -> dict[str, object]:
    """Everything above in one dict, ready to serialise next to the raw report."""
    return {
        "ap_clean": round(report.ap_clean, 4),
        "mpc": round(mpc(report), 4),
        "rpc": round(rpc(report), 4),
        "resilience_rate": {key: round(value, 4) for key, value in resilience_rate(report).items()},
        "robustness_accuracy": {
            key: round(value, 4) for key, value in robustness_accuracy(report).items()
        },
        "category_scores": {key: round(value, 4) for key, value in category_scores(report).items()},
        "robust_score_plan": round(robust_score(report), 2),
        "robust_score_normalized": round(robust_score(report, normalize=True), 2),
        "covered_categories": covered_categories(report),
        "severity_monotonicity": severity_monotonicity(report),
        "normalize_score": normalize_score,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else 0.0
