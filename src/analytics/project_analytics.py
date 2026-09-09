"""Project-level analytics: trends and comparisons across the saved runs of one project.

Like :mod:`src.analytics.run_analytics`, everything here is a pure function over
dictionaries — no database or filesystem I/O. The API layer is responsible for
collecting each project's run items (run_id, report, optional display metadata)
and passing them in.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.analytics.run_analytics import (
    compute_run_attacks_breakdown,
    compute_run_classes_breakdown,
    compute_run_severity_breakdown,
    compute_run_summary,
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _trend_direction(scores: list[float]) -> str:
    """Compare the first and last robust score; small changes count as stable."""
    if len(scores) < 2:
        return "insufficient_data"
    delta = scores[-1] - scores[0]
    if delta > 1.0:
        return "improving"
    if delta < -1.0:
        return "declining"
    return "stable"


def compute_project_analytics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate every completed run of a project into decision-level analytics.

    Args:
        items: One dict per saved run: ``{"run_id", "report", "name", "created_at"}``.
            Items without a report (failed/pending runs) are skipped.

    Returns:
        Dictionary with per-run table rows, robustness trend, overall means,
        cross-run worst attack / most sensitive severity / most vulnerable
        class, and an object-level failure summary.
    """
    run_rows: list[dict[str, Any]] = []
    attack_deg_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    severity_deg: dict[int, list[dict[str, Any]]] = defaultdict(list)
    class_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total_ground_truth_objects": 0,
            "clean_detected_count": 0,
            "attacked_detected_count": 0,
            "lost_objects_count": 0,
            "hallucinated_objects_count": 0,
        }
    )

    for item in items:
        report = item.get("report")
        if not isinstance(report, dict):
            continue
        summary = compute_run_summary(report)
        worst = summary.get("worst_attack") or {}
        run_rows.append(
            {
                "run_id": item.get("run_id") or summary.get("run_id", ""),
                "name": item.get("name"),
                "created_at": item.get("created_at"),
                "task_id": summary.get("task_id"),
                "metric_key": summary.get("primary_metric", {}).get("key"),
                "metric_label": summary.get("primary_metric", {}).get("label"),
                "clean": summary.get("ap_clean"),
                "attacked_mean": summary.get("mean_attack_metric"),
                "degradation_percent": summary.get("overall_degradation_percent"),
                "robust_score": summary.get("robust_score"),
                "worst_attack": worst.get("attack"),
                "n_cells": summary.get("total_cells_evaluated", 0),
                "n_samples": summary.get("n_samples", 0),
                "data_state": summary.get("data_state"),
            }
        )
        for breakdown in compute_run_attacks_breakdown(report):
            if breakdown.get("mean_degradation_percent") is not None:
                attack_deg_by_name[breakdown["attack"]].append(
                    {
                        "degradation_percent": breakdown["mean_degradation_percent"],
                        "worst_severity": breakdown.get("worst_severity"),
                        "run_id": run_rows[-1]["run_id"],
                    }
                )
        for severity_row in compute_run_severity_breakdown(report):
            if severity_row.get("mean_degradation_percent") is not None:
                severity_deg[int(severity_row["severity"])].append(
                    {"degradation_percent": severity_row["mean_degradation_percent"]}
                )
        for class_row in compute_run_classes_breakdown(report):
            stats = class_stats[class_row["class_name"]]
            for key in stats:
                stats[key] += int(class_row.get(key, 0) or 0)

    ordered_runs = sorted(run_rows, key=lambda row: (row.get("created_at") or "", row["run_id"]))
    robust_scores = [row["robust_score"] for row in ordered_runs if row["robust_score"] is not None]

    worst_attack_overall = None
    attack_candidates: list[dict[str, Any]] = []
    for attack_name, occurrences in attack_deg_by_name.items():
        degradations = [entry["degradation_percent"] for entry in occurrences]
        worst_entry = max(occurrences, key=lambda entry: entry["degradation_percent"])
        attack_candidates.append(
            {
                "attack": attack_name,
                "runs_affected": len(occurrences),
                "mean_degradation_percent": _round(_mean(degradations), 2),
                "max_degradation_percent": _round(max(degradations), 2),
                "worst_severity": worst_entry.get("worst_severity"),
                "worst_run_id": worst_entry.get("run_id"),
            }
        )
    if attack_candidates:
        worst_attack_overall = max(attack_candidates, key=lambda item: item["mean_degradation_percent"])

    severity_rows = [
        {
            "severity": severity,
            "runs_measured": len(occurrences),
            "mean_degradation_percent": _round(_mean([e["degradation_percent"] for e in occurrences]), 2),
        }
        for severity, occurrences in sorted(severity_deg.items())
    ]
    most_sensitive_severity = (
        max(severity_rows, key=lambda item: item["mean_degradation_percent"]) if severity_rows else None
    )

    class_rows: list[dict[str, Any]] = []
    for class_name, stats in sorted(class_stats.items()):
        gt = stats["total_ground_truth_objects"]
        clean_rate = stats["clean_detected_count"] / gt if gt > 0 else 0.0
        attacked_rate = stats["attacked_detected_count"] / gt if gt > 0 else 0.0
        drop = max(0.0, (clean_rate - attacked_rate) / clean_rate) if clean_rate > 0.0 else 0.0
        class_rows.append(
            {
                "class_name": class_name,
                "total_ground_truth_objects": gt,
                "clean_detection_rate": _round(clean_rate),
                "attacked_detection_rate": _round(attacked_rate),
                "detection_drop_percent": _round(drop * 100.0, 2),
                "lost_objects_count": stats["lost_objects_count"],
                "hallucinated_objects_count": stats["hallucinated_objects_count"],
            }
        )
    most_vulnerable_class = (
        max(class_rows, key=lambda item: (item["detection_drop_percent"], item["total_ground_truth_objects"]))
        if class_rows
        else None
    )

    measured = [row for row in ordered_runs if row["data_state"] == "MEASURED"]
    overall = {
        "run_count": len(ordered_runs),
        "measured_run_count": len(measured),
        "mean_clean": _round(_mean([row["clean"] for row in measured if row["clean"] is not None])),
        "mean_attacked": _round(_mean([row["attacked_mean"] for row in measured if row["attacked_mean"] is not None])),
        "mean_degradation_percent": _round(
            _mean([row["degradation_percent"] for row in measured if row["degradation_percent"] is not None]), 2
        ),
        "mean_robust_score": _round(_mean(robust_scores), 2),
        "total_cells": sum(row["n_cells"] for row in ordered_runs),
        "total_samples_runs": sum(row["n_samples"] for row in ordered_runs),
        "data_state": "MEASURED" if measured else "NO_DATA",
    }

    return {
        "runs": ordered_runs,
        "trend": {
            "robust_scores": robust_scores,
            "direction": _trend_direction(robust_scores),
            "delta_first_last": _round(robust_scores[-1] - robust_scores[0], 2) if len(robust_scores) >= 2 else None,
        },
        "overall": overall,
        "worst_attack_overall": worst_attack_overall,
        "severity_sensitivity": severity_rows,
        "most_sensitive_severity": most_sensitive_severity,
        "class_vulnerability": class_rows,
        "most_vulnerable_class": most_vulnerable_class,
    }


def compute_runs_comparison(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an N-run comparison matrix (attack × run and severity × run).

    Args:
        items: Same shape as :func:`compute_project_analytics` (usually a
            subset of runs the user explicitly selected).

    Returns:
        Dictionary with per-run summary rows, per-attack and per-severity
        pivots keyed by run_id, and a flat cell list for heatmap rendering.
    """
    run_rows: list[dict[str, Any]] = []
    attack_pivot: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    severity_pivot: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    flat_cells: list[dict[str, Any]] = []
    task_ids: set[str] = set()

    for item in items:
        report = item.get("report")
        if not isinstance(report, dict):
            continue
        run_id = item.get("run_id") or report.get("run_id", "")
        summary = compute_run_summary(report)
        task_id = summary.get("task_id")
        if task_id:
            task_ids.add(task_id)
        metric = summary.get("primary_metric", {})
        run_rows.append(
            {
                "run_id": run_id,
                "name": item.get("name"),
                "created_at": item.get("created_at"),
                "task_id": task_id,
                "metric_key": metric.get("key"),
                "metric_label": metric.get("label"),
                "clean": summary.get("ap_clean"),
                "attacked_mean": summary.get("mean_attack_metric"),
                "degradation_percent": summary.get("overall_degradation_percent"),
                "robust_score": summary.get("robust_score"),
                "data_state": summary.get("data_state"),
            }
        )
        for breakdown in compute_run_attacks_breakdown(report):
            attack = breakdown["attack"]
            entry = {
                "mean_metric": breakdown.get("mean_metric"),
                "mean_degradation_percent": breakdown.get("mean_degradation_percent"),
            }
            if entry["mean_degradation_percent"] is not None:
                attack_pivot[attack][run_id] = entry
            for severity, value in (breakdown.get("degradation_percent_by_severity") or {}).items():
                flat_cells.append(
                    {
                        "run_id": run_id,
                        "attack": attack,
                        "severity": int(severity),
                        "degradation_percent": value,
                    }
                )
        for severity_row in compute_run_severity_breakdown(report):
            if severity_row.get("mean_degradation_percent") is None:
                continue
            severity_pivot[int(severity_row["severity"])][run_id] = {
                "mean_metric": severity_row.get("mean_metric"),
                "mean_degradation_percent": severity_row.get("mean_degradation_percent"),
                "worst_attack": severity_row.get("worst_attack"),
            }

    ordered_runs = sorted(run_rows, key=lambda row: (row.get("created_at") or "", row["run_id"]))
    run_ids = [row["run_id"] for row in ordered_runs]
    metric_row = next((row for row in ordered_runs if row["metric_key"]), None)

    return {
        "run_ids": run_ids,
        "metric_key": metric_row["metric_key"] if metric_row else None,
        "metric_label": metric_row["metric_label"] if metric_row else None,
        "comparable": len(task_ids) <= 1,
        "task_ids": sorted(task_ids),
        "runs": ordered_runs,
        "attack_pivot": {
            attack: {run_id: pivot.get(run_id) for run_id in run_ids}
            for attack, pivot in sorted(attack_pivot.items())
        },
        "severity_pivot": {
            severity: {run_id: pivot.get(run_id) for run_id in run_ids}
            for severity, pivot in sorted(severity_pivot.items())
        },
        "cells": flat_cells,
    }
