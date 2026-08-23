"""Pure analytics aggregation functions for model comparison records.

Operates on comparison records and optional associated baseline/candidate run reports.
Computes recovery curves, class migrations, and granular failure state transitions
(e.g., CORRECT -> FAILED, FAILED -> RECOVERED, FAILED -> STILL_FAILED).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.analytics.run_analytics import compute_run_classes_breakdown


def compute_comparison_summary(
    comparison: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    candidate_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute executive-level paired comparison summary between baseline and candidate models.

    Args:
        comparison: Serialized model comparison record.
        baseline_run: Optional baseline RunJob record with completed report.
        candidate_run: Optional candidate RunJob record with completed report.

    Returns:
        Dictionary containing paired status, clean/attacked score deltas, recovery rates,
        and gate verdict determination.
    """
    paired = bool(comparison.get("paired", False))
    incompatibilities = list(comparison.get("incompatibilities", []))
    recovery_data = comparison.get("recovery_report") or {}
    recovery_rate_data = recovery_data.get("recovery_rate") or {}

    base_report = (baseline_run or {}).get("report") or {}
    cand_report = (candidate_run or {}).get("report") or {}

    baseline_clean = float(
        recovery_data.get("baseline_clean")
        or base_report.get("ap_clean", 0.0)
    )
    candidate_clean = float(
        recovery_data.get("candidate_clean")
        or cand_report.get("ap_clean", 0.0)
    )
    clean_delta = candidate_clean - baseline_clean

    base_attacked = _calculate_mean_attack_score(base_report)
    cand_attacked = _calculate_mean_attack_score(cand_report)
    attacked_delta = cand_attacked - base_attacked

    recovery_ratio = recovery_rate_data.get("ratio_value")
    if recovery_ratio is None and paired:
        lost = baseline_clean - base_attacked
        gained = cand_attacked - base_attacked
        recovery_ratio = (gained / lost) if lost > 0 else None

    recovery_percent = (
        round(recovery_ratio * 100.0, 4) if recovery_ratio is not None else None
    )

    clean_retention_ratio = (
        candidate_clean / baseline_clean if baseline_clean > 0 else 1.0
    )

    verdict, reasons = _determine_comparison_verdict(
        paired=paired,
        incompatibilities=incompatibilities,
        clean_delta=clean_delta,
        attacked_delta=attacked_delta,
        recovery_ratio=recovery_ratio,
        baseline_clean=baseline_clean,
    )

    return {
        "comparison_id": str(comparison.get("comparison_id", "")),
        "baseline_run_id": str(comparison.get("baseline_run_id", "")),
        "candidate_run_id": str(comparison.get("candidate_run_id", "")),
        "paired": paired,
        "incompatibilities": incompatibilities,
        "baseline_clean_ap": round(baseline_clean, 4),
        "candidate_clean_ap": round(candidate_clean, 4),
        "clean_ap_delta": round(clean_delta, 4),
        "baseline_attacked_ap": round(base_attacked, 4),
        "candidate_attacked_ap": round(cand_attacked, 4),
        "attacked_ap_delta": round(attacked_delta, 4),
        "recovery_rate": {
            "ratio_value": round(recovery_ratio, 6) if recovery_ratio is not None else None,
            "percent_value": recovery_percent,
            "unit": "percent",
        },
        "clean_retention_ratio": round(clean_retention_ratio, 4),
        "verdict": verdict,
        "verdict_reasons": reasons,
        "metric_deltas": dict(comparison.get("metric_deltas", {})),
    }


def compute_comparison_recovery(
    comparison: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    candidate_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute granular recovery analytics broken down per attack and per severity.

    Args:
        comparison: Serialized model comparison record.
        baseline_run: Optional baseline RunJob record with completed report.
        candidate_run: Optional candidate RunJob record with completed report.

    Returns:
        Dictionary containing overall, per-attack, and per-severity recovery metrics.
    """
    base_report = (baseline_run or {}).get("report") or {}
    cand_report = (candidate_run or {}).get("report") or {}

    baseline_clean = float(base_report.get("ap_clean", 0.0))
    candidate_clean = float(cand_report.get("ap_clean", 0.0))

    base_cells = {
        (str(c.get("attack")), int(c.get("severity", 1))): float(c.get("ap", 0.0))
        for c in base_report.get("cells", [])
    }
    cand_cells = {
        (str(c.get("attack")), int(c.get("severity", 1))): float(c.get("ap", 0.0))
        for c in cand_report.get("cells", [])
    }

    attack_names = sorted({key[0] for key in set(base_cells.keys()) | set(cand_cells.keys())})
    per_attack_recovery: list[dict[str, Any]] = []

    for attack in attack_names:
        base_aps = [val for key, val in base_cells.items() if key[0] == attack]
        cand_aps = [val for key, val in cand_cells.items() if key[0] == attack]

        base_mean_ap = sum(base_aps) / len(base_aps) if base_aps else 0.0
        cand_mean_ap = sum(cand_aps) / len(cand_aps) if cand_aps else 0.0
        delta_ap = cand_mean_ap - base_mean_ap

        lost_ap = baseline_clean - base_mean_ap
        if lost_ap > 0.0:
            rec_ratio = delta_ap / lost_ap
            rec_percent = rec_ratio * 100.0
        else:
            rec_ratio = None
            rec_percent = None

        if rec_ratio is not None and rec_ratio >= 0.8:
            status = "RECOVERED"
        elif rec_ratio is not None and rec_ratio > 0.0:
            status = "PARTIAL_RECOVERY"
        elif delta_ap < 0.0:
            status = "REGRESSED"
        else:
            status = "UNCHANGED"

        group = "A"
        for c in base_report.get("cells", []):
            if c.get("attack") == attack:
                group = str(c.get("group", "A"))
                break

        per_attack_recovery.append({
            "attack": attack,
            "group": group,
            "baseline_mean_ap": round(base_mean_ap, 4),
            "candidate_mean_ap": round(cand_mean_ap, 4),
            "delta_ap": round(delta_ap, 4),
            "recovery_ratio": round(rec_ratio, 6) if rec_ratio is not None else None,
            "recovery_percent": round(rec_percent, 4) if rec_percent is not None else None,
            "status": status,
        })

    # Per-severity recovery
    severity_levels = sorted({key[1] for key in set(base_cells.keys()) | set(cand_cells.keys())})
    per_severity_recovery: list[dict[str, Any]] = []

    for sev in severity_levels:
        base_aps = [val for key, val in base_cells.items() if key[1] == sev]
        cand_aps = [val for key, val in cand_cells.items() if key[1] == sev]

        base_mean = sum(base_aps) / len(base_aps) if base_aps else 0.0
        cand_mean = sum(cand_aps) / len(cand_aps) if cand_aps else 0.0
        delta = cand_mean - base_mean

        lost = baseline_clean - base_mean
        ratio = (delta / lost) if lost > 0 else None

        per_severity_recovery.append({
            "severity": sev,
            "baseline_mean_ap": round(base_mean, 4),
            "candidate_mean_ap": round(cand_mean, 4),
            "delta_ap": round(delta, 4),
            "recovery_ratio": round(ratio, 6) if ratio is not None else None,
            "recovery_percent": round(ratio * 100.0, 4) if ratio is not None else None,
        })

    base_attacked_total = _calculate_mean_attack_score(base_report)
    cand_attacked_total = _calculate_mean_attack_score(cand_report)
    overall_lost = baseline_clean - base_attacked_total
    overall_ratio = (
        (cand_attacked_total - base_attacked_total) / overall_lost
        if overall_lost > 0
        else None
    )

    return {
        "overall_recovery": {
            "ratio_value": round(overall_ratio, 6) if overall_ratio is not None else None,
            "percent_value": (
                round(overall_ratio * 100.0, 4) if overall_ratio is not None else None
            ),
            "unit": "percent",
            "formula": "(candidate_attacked - baseline_attacked) / (baseline_clean - baseline_attacked)",
        },
        "per_attack_recovery": per_attack_recovery,
        "per_severity_recovery": per_severity_recovery,
        "clean_retention": {
            "baseline_clean": round(baseline_clean, 4),
            "candidate_clean": round(candidate_clean, 4),
            "clean_delta": round(candidate_clean - baseline_clean, 4),
            "retained_percent": (
                round(candidate_clean / baseline_clean * 100.0, 4)
                if baseline_clean > 0
                else 100.0
            ),
        },
    }


def compute_comparison_classes(
    comparison: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    candidate_run: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute per-class performance comparisons and recovery deltas.

    Args:
        comparison: Serialized model comparison record.
        baseline_run: Optional baseline RunJob record with completed report.
        candidate_run: Optional candidate RunJob record with completed report.

    Returns:
        List of per-class comparison records with baseline vs candidate detection rates.
    """
    base_report = (baseline_run or {}).get("report") or {}
    cand_report = (candidate_run or {}).get("report") or {}

    base_classes = {item["class_name"]: item for item in compute_run_classes_breakdown(base_report)}
    cand_classes = {item["class_name"]: item for item in compute_run_classes_breakdown(cand_report)}

    all_classes = sorted(set(base_classes.keys()) | set(cand_classes.keys()))
    results: list[dict[str, Any]] = []

    for cls_name in all_classes:
        base_item = base_classes.get(cls_name, {})
        cand_item = cand_classes.get(cls_name, {})

        base_clean_rate = float(base_item.get("clean_detection_rate", 0.0))
        cand_clean_rate = float(cand_item.get("clean_detection_rate", 0.0))
        clean_delta = cand_clean_rate - base_clean_rate

        base_attacked_rate = float(base_item.get("attacked_detection_rate", 0.0))
        cand_attacked_rate = float(cand_item.get("attacked_detection_rate", 0.0))
        attacked_delta = cand_attacked_rate - base_attacked_rate

        lost_rate = base_clean_rate - base_attacked_rate
        if lost_rate > 0.0:
            rec_ratio = attacked_delta / lost_rate
            rec_percent = rec_ratio * 100.0
        else:
            rec_ratio = None
            rec_percent = None

        if attacked_delta > 0.05:
            status = "IMPROVED"
        elif attacked_delta < -0.05:
            status = "REGRESSED"
        else:
            status = "NEUTRAL"

        results.append({
            "class_name": cls_name,
            "baseline_clean_rate": round(base_clean_rate, 4),
            "candidate_clean_rate": round(cand_clean_rate, 4),
            "clean_delta": round(clean_delta, 4),
            "baseline_attacked_rate": round(base_attacked_rate, 4),
            "candidate_attacked_rate": round(cand_attacked_rate, 4),
            "attacked_delta": round(attacked_delta, 4),
            "recovery_ratio": round(rec_ratio, 6) if rec_ratio is not None else None,
            "recovery_percent": round(rec_percent, 4) if rec_percent is not None else None,
            "status": status,
        })

    return results


def compute_comparison_failures(
    comparison: Mapping[str, Any],
    baseline_run: Mapping[str, Any] | None = None,
    candidate_run: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute detailed failure state transitions between baseline and candidate runs.

    Transitions categorized as:
      - FAILED  -> RECOVERED
      - FAILED  -> STILL_FAILED
      - CORRECT -> FAILED (regressions)

    Args:
        comparison: Serialized model comparison record.
        baseline_run: Optional baseline RunJob record with completed report.
        candidate_run: Optional candidate RunJob record with completed report.

    Returns:
        Dictionary containing failure counts, transition lists, and net recovery rates.
    """
    base_report = (baseline_run or {}).get("report") or {}
    cand_report = (candidate_run or {}).get("report") or {}

    base_samples = {
        _sample_key(s): s for s in base_report.get("sample_results", [])
    }
    cand_samples = {
        _sample_key(s): s for s in cand_report.get("sample_results", [])
    }

    # Fallback to worst_cases if sample_results are empty
    if not base_samples and base_report.get("worst_cases"):
        base_samples = {_sample_key(s): s for s in base_report.get("worst_cases", [])}
    if not cand_samples and cand_report.get("worst_cases"):
        cand_samples = {_sample_key(s): s for s in cand_report.get("worst_cases", [])}

    all_keys = sorted(set(base_samples.keys()) | set(cand_samples.keys()))

    recovered: list[dict[str, Any]] = []
    still_failed: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []

    for key in all_keys:
        base_sample = base_samples.get(key)
        cand_sample = cand_samples.get(key)

        base_failed = _is_sample_failed(base_sample)
        cand_failed = _is_sample_failed(cand_sample)

        entry = {
            "key": key,
            "sample_id": (base_sample or cand_sample or {}).get("sample_id", key),
            "attack": (base_sample or cand_sample or {}).get("attack", ""),
            "severity": (base_sample or cand_sample or {}).get("severity", 1),
            "baseline_degradation": round(float((base_sample or {}).get("degradation_hint", 0.0)), 6),
            "candidate_degradation": round(float((cand_sample or {}).get("degradation_hint", 0.0)), 6),
        }

        if base_failed and not cand_failed:
            recovered.append({**entry, "transition": "FAILED->RECOVERED"})
        elif base_failed and cand_failed:
            still_failed.append({**entry, "transition": "FAILED->STILL_FAILED"})
        elif not base_failed and cand_failed:
            regressed.append({**entry, "transition": "CORRECT->FAILED"})

    base_fail_count = len(recovered) + len(still_failed)
    cand_fail_count = len(still_failed) + len(regressed)
    net_reduction = base_fail_count - cand_fail_count
    fail_recovery_rate = (len(recovered) / base_fail_count) if base_fail_count > 0 else 0.0

    return {
        "comparison_id": str(comparison.get("comparison_id", "")),
        "total_baseline_failures": base_fail_count,
        "total_candidate_failures": cand_fail_count,
        "recovered_count": len(recovered),
        "still_failed_count": len(still_failed),
        "regressed_count": len(regressed),
        "net_failure_reduction": net_reduction,
        "failure_recovery_rate": round(fail_recovery_rate, 4),
        "recovered_failures": recovered,
        "persistent_failures": still_failed,
        "new_regressed_failures": regressed,
    }


def _calculate_mean_attack_score(report: Mapping[str, Any]) -> float:
    """Helper to compute mean attacked score across all report cells."""
    cells = list(report.get("cells", []))
    if not cells:
        return float(report.get("ap_clean", 0.0))
    return sum(float(cell.get("ap", 0.0)) for cell in cells) / len(cells)


def _sample_key(sample: Mapping[str, Any]) -> str:
    return f"{sample.get('sample_id', '')}::{sample.get('attack', '')}::{sample.get('severity', 1)}"


def _is_sample_failed(sample: Mapping[str, Any] | None) -> bool:
    if sample is None:
        return False
    hint = float(sample.get("degradation_hint", 0.0))
    if hint > 0.0:
        return True
    if sample.get("failed") is True:
        return True
    return False


def _determine_comparison_verdict(
    *,
    paired: bool,
    incompatibilities: list[str],
    clean_delta: float,
    attacked_delta: float,
    recovery_ratio: float | None,
    baseline_clean: float,
) -> tuple[str, list[str]]:
    """Determine promotional verdict based on safety and recovery gates."""
    reasons: list[str] = []

    if not paired:
        return "INCOMPATIBLE_PROTOCOL", [
            f"Runs are not paired on protocol parameters: {', '.join(incompatibilities)}"
        ]

    # Clean degradation check (plan allows max 2% clean drop)
    max_clean_drop = 0.02 * baseline_clean if baseline_clean > 0 else 0.02
    if clean_delta < -max_clean_drop:
        reasons.append(f"Clean accuracy regressed by {abs(clean_delta):.4f} (> {max_clean_drop:.4f})")
        return "REGRESSED_CLEAN", reasons

    if attacked_delta < -0.01:
        reasons.append(f"Performance under attack regressed by {abs(attacked_delta):.4f}")
        return "REGRESSED_ATTACK", reasons

    if recovery_ratio is not None and recovery_ratio >= 0.15:
        reasons.append(f"Model recovered {recovery_ratio * 100:.1f}% of damage under attack")
        return "PROMOTABLE", reasons

    reasons.append("Model did not demonstrate significant robustness gain (recovery < 15%)")
    return "NEUTRAL", reasons
