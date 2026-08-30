"""Pure analytics aggregation functions for individual test run reports.

All functions in this module operate on dictionary representations of RunReport
(as produced by RunReport.as_dict() or retrieved from the database store).
No filesystem I/O or external network queries are performed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

_TASK_METRICS: dict[str, tuple[str, str]] = {
    "detection2d": ("map50_95", "mAP@50-95"),
    "segmentation": ("miou", "mIoU"),
    "detection3d": ("kitti_3d_ap", "KITTI 3D AP"),
}


def _task_id(report: Mapping[str, Any]) -> str:
    provenance = report.get("provenance") or {}
    run_config = provenance.get("run_config") or {}
    explicit = run_config.get("task_id") or report.get("task_id") or report.get("task")
    if explicit in _TASK_METRICS:
        return str(explicit)
    clean = (report.get("metrics") or {}).get("clean") or {}
    if "kitti_3d_ap" in clean:
        return "detection3d"
    if "miou" in clean:
        return "segmentation"
    return "detection2d"


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _cell_primary_value(
    cell: Mapping[str, Any], task_id: str, metric_key: str, *, allow_legacy_ap: bool = True
) -> float | None:
    value = _finite_number((cell.get("metrics") or {}).get(metric_key))
    if value is None and allow_legacy_ap and task_id != "segmentation":
        value = _finite_number(cell.get("ap"))
    return value


def _decision_context(report: Mapping[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    task_id = _task_id(report)
    metric_key, metric_label = _TASK_METRICS[task_id]
    metrics = report.get("metrics") or {}
    clean_metrics = metrics.get("clean") or {}
    benchmark_available = bool(
        report.get("benchmark_metrics_available", metrics.get("benchmark_metrics_available", True))
    )
    clean_value = _finite_number(clean_metrics.get(metric_key))
    allow_legacy_ap = clean_value is None and task_id != "segmentation"
    if clean_value is None and benchmark_available and task_id != "segmentation":
        clean_value = _finite_number(report.get("ap_clean"))
    attacked_values = [
        value
        for cell in cells
        if (value := _cell_primary_value(cell, task_id, metric_key, allow_legacy_ap=allow_legacy_ap)) is not None
    ]
    attacked_mean = sum(attacked_values) / len(attacked_values) if attacked_values else None
    data_state = (
        "MEASURED" if benchmark_available and clean_value is not None and attacked_mean is not None else "NO_DATA"
    )
    if not benchmark_available:
        data_state = "NO_GROUND_TRUTH"
        clean_value = None
        attacked_mean = None

    provenance = report.get("provenance") or {}
    run_config = provenance.get("run_config") or {}
    limitations: list[str] = []
    if not benchmark_available:
        limitations.append("Ground-truth annotations are unavailable; benchmark metrics cannot be computed.")
    if report.get("simulation_only", True):
        limitations.append("Simulation-only evidence does not validate production safety.")
    n_samples = int(report.get("n_samples", 0) or 0)
    if benchmark_available and 0 < n_samples < 30:
        limitations.append("Small evaluation sample; uncertainty may be high and results should not gate deployment.")

    return {
        "task_id": task_id,
        "primary_metric": {
            "key": metric_key,
            "label": metric_label,
            "unit": "ratio",
            "clean": None if clean_value is None else round(clean_value, 6),
            "attacked_mean": None if attacked_mean is None else round(attacked_mean, 6),
            "value": None if attacked_mean is None else round(attacked_mean, 6),
        },
        "data_state": data_state,
        "protocol": {
            "benchmark_protocol_id": provenance.get("benchmark_protocol_id") or run_config.get("benchmark_protocol_id"),
            "dataset_version_id": provenance.get("dataset_version_id") or run_config.get("dataset_version_id"),
            "split": provenance.get("split")
            or run_config.get("split")
            or (run_config.get("dataset_params") or {}).get("split"),
            "checkpoint_hash": (provenance.get("model") or {}).get("checkpoint_hash"),
            "seed": run_config.get("seed"),
            "iou_threshold": run_config.get("iou_threshold"),
        },
        "limitations": limitations,
        "benchmark_metrics_available": benchmark_available,
    }


def compute_run_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compute high-level executive analytics for a single benchmark run.

    Args:
        report: Serialized RunReport dictionary.

    Returns:
        Dictionary containing overall performance, group-level breakdown,
        worst/resilient attack indicators, and robust score metrics.
    """
    cells = list(report.get("cells", []))
    skipped = list(report.get("skipped", []))
    metrics = dict(report.get("metrics", {}))
    decision = _decision_context(report, cells)
    clean_value = decision["primary_metric"]["clean"]
    attacked_mean = decision["primary_metric"]["attacked_mean"]
    metric_key = decision["primary_metric"]["key"]
    metric_label = decision["primary_metric"]["label"]
    task_id = decision["task_id"]
    allow_legacy_ap = _finite_number((metrics.get("clean") or {}).get(metric_key)) is None

    if not cells:
        return {
            "run_id": report.get("run_id", ""),
            "model": report.get("model", ""),
            "model_version": report.get("model_version", ""),
            "dataset": report.get("dataset", ""),
            "n_samples": int(report.get("n_samples", 0)),
            "ap_clean": clean_value,
            "mean_attack_ap": None,
            "mean_attack_metric": None,
            "overall_degradation_ratio": None,
            "overall_degradation_percent": None,
            "robust_score": None,
            "worst_attack": None,
            "most_resilient_attack": None,
            "group_summaries": {},
            "total_cells_evaluated": 0,
            "total_attacks_evaluated": 0,
            "total_skipped_attacks": len(skipped),
            "clean_metrics": metrics.get("clean", {}),
            "seconds": float(report.get("seconds", 0.0)),
            **decision,
        }

    attack_cells_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_cells_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cell in cells:
        attack_name = str(cell.get("attack", ""))
        group_name = str(cell.get("group", "A"))
        attack_cells_map[attack_name].append(cell)
        group_cells_map[group_name].append(cell)

    overall_degradation_ratio = (
        max(0.0, (clean_value - attacked_mean) / clean_value)
        if clean_value is not None and clean_value > 0.0 and attacked_mean is not None
        else None
    )

    attack_summaries = _aggregate_attack_metrics(
        attack_cells_map,
        clean_value,
        task_id=task_id,
        metric_key=metric_key,
        metric_label=metric_label,
        allow_legacy_ap=allow_legacy_ap,
    )
    sorted_by_degradation = sorted(
        (item for item in attack_summaries if item["mean_degradation_ratio"] is not None),
        key=lambda item: item["mean_degradation_ratio"],
        reverse=True,
    )

    worst_attack = sorted_by_degradation[0] if sorted_by_degradation else None
    most_resilient_attack = sorted_by_degradation[-1] if sorted_by_degradation else None

    group_summaries: dict[str, dict[str, Any]] = {}
    for group_name, group_cells in sorted(group_cells_map.items()):
        group_values = [
            value
            for cell in group_cells
            if (value := _cell_primary_value(cell, task_id, metric_key, allow_legacy_ap=allow_legacy_ap)) is not None
        ]
        group_mean_ap = sum(group_values) / len(group_values) if group_values else None
        group_degradation_ratio = (
            max(0.0, (clean_value - group_mean_ap) / clean_value)
            if clean_value is not None and clean_value > 0.0 and group_mean_ap is not None
            else None
        )
        group_attacks = sorted({str(cell.get("attack", "")) for cell in group_cells})
        group_summaries[group_name] = {
            "cell_count": len(group_cells),
            "metric_key": metric_key,
            "metric_label": metric_label,
            "mean_ap": None if group_mean_ap is None else round(group_mean_ap, 4),
            "mean_metric": None if group_mean_ap is None else round(group_mean_ap, 4),
            "mean_degradation_ratio": None if group_degradation_ratio is None else round(group_degradation_ratio, 6),
            "mean_degradation_percent": None
            if group_degradation_ratio is None
            else round(group_degradation_ratio * 100.0, 4),
            "attacks": group_attacks,
        }

    robust_score = None if overall_degradation_ratio is None else max(0.0, 100.0 * (1.0 - overall_degradation_ratio))

    return {
        "run_id": report.get("run_id", ""),
        "model": report.get("model", ""),
        "model_version": report.get("model_version", ""),
        "dataset": report.get("dataset", ""),
        "n_samples": int(report.get("n_samples", 0)),
        "ap_clean": clean_value,
        "mean_attack_ap": attacked_mean,
        "mean_attack_metric": attacked_mean,
        "overall_degradation_ratio": None if overall_degradation_ratio is None else round(overall_degradation_ratio, 6),
        "overall_degradation_percent": None
        if overall_degradation_ratio is None
        else round(overall_degradation_ratio * 100.0, 4),
        "robust_score": None if robust_score is None else round(robust_score, 2),
        "worst_attack": worst_attack,
        "most_resilient_attack": most_resilient_attack,
        "group_summaries": group_summaries,
        "total_cells_evaluated": len(cells),
        "total_attacks_evaluated": len(attack_cells_map),
        "total_skipped_attacks": len(skipped),
        "clean_metrics": metrics.get("clean", {}),
        "seconds": float(report.get("seconds", 0.0)),
        **decision,
    }


def compute_run_attacks_breakdown(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute detailed per-attack robustness analytics across all evaluated severities.

    Args:
        report: Serialized RunReport dictionary.

    Returns:
        List of per-attack breakdown dictionaries ranked from most degraded to least degraded.
    """
    cells = list(report.get("cells", []))
    sample_results = list(report.get("sample_results", []))

    if not cells:
        return []

    attack_cells_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        attack_cells_map[str(cell.get("attack", ""))].append(cell)

    attack_failure_counts: dict[str, int] = defaultdict(int)
    for sample in sample_results:
        degradation_hint = float(sample.get("degradation_hint", 0.0))
        if degradation_hint > 0.0 or sample.get("failed") is True:
            attack_failure_counts[str(sample.get("attack", ""))] += 1

    decision = _decision_context(report, cells)
    clean_value = decision["primary_metric"]["clean"]
    metric_key = decision["primary_metric"]["key"]
    breakdowns = _aggregate_attack_metrics(
        attack_cells_map,
        clean_value,
        failure_counts=attack_failure_counts,
        task_id=decision["task_id"],
        metric_key=metric_key,
        metric_label=decision["primary_metric"]["label"],
        allow_legacy_ap=_finite_number(((report.get("metrics") or {}).get("clean") or {}).get(metric_key)) is None,
    )
    return sorted(
        breakdowns,
        key=lambda item: item["mean_degradation_ratio"] if item["mean_degradation_ratio"] is not None else -1.0,
        reverse=True,
    )


def compute_run_severity_breakdown(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute robustness analytics broken down by corruption/perturbation severity level (1..5).

    Args:
        report: Serialized RunReport dictionary.

    Returns:
        List of per-severity summary dictionaries sorted in ascending order of severity level.
    """
    cells = list(report.get("cells", []))

    if not cells:
        return []

    decision = _decision_context(report, cells)
    clean_ap = decision["primary_metric"]["clean"]
    task_id = decision["task_id"]
    metric_key = decision["primary_metric"]["key"]
    metric_label = decision["primary_metric"]["label"]
    allow_legacy_ap = _finite_number(((report.get("metrics") or {}).get("clean") or {}).get(metric_key)) is None

    severity_cells_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        sev = int(cell.get("severity", 1))
        severity_cells_map[sev].append(cell)

    results: list[dict[str, Any]] = []
    for severity_level, sev_cells in sorted(severity_cells_map.items()):
        aps = [
            value
            for cell in sev_cells
            if (value := _cell_primary_value(cell, task_id, metric_key, allow_legacy_ap=allow_legacy_ap)) is not None
        ]
        mean_ap = sum(aps) / len(aps) if aps else None
        min_ap = min(aps) if aps else None
        max_ap = max(aps) if aps else None
        degradation_ratio = (
            max(0.0, (clean_ap - mean_ap) / clean_ap)
            if clean_ap is not None and clean_ap > 0.0 and mean_ap is not None
            else None
        )

        scored_cells = [
            (cell, value)
            for cell in sev_cells
            if (value := _cell_primary_value(cell, task_id, metric_key, allow_legacy_ap=allow_legacy_ap)) is not None
        ]
        worst_cell, worst_attack_ap = min(scored_cells, key=lambda item: item[1]) if scored_cells else (None, None)
        worst_attack_name = str(worst_cell.get("attack", "")) if worst_cell else None
        worst_attack_degradation = (
            max(0.0, (clean_ap - worst_attack_ap) / clean_ap * 100.0)
            if clean_ap is not None and clean_ap > 0.0 and worst_attack_ap is not None
            else None
        )

        results.append(
            {
                "severity": severity_level,
                "metric_key": metric_key,
                "metric_label": metric_label,
                "evaluated_cells_count": len(sev_cells),
                "mean_ap": None if mean_ap is None else round(mean_ap, 4),
                "mean_metric": None if mean_ap is None else round(mean_ap, 4),
                "mean_degradation_ratio": None if degradation_ratio is None else round(degradation_ratio, 6),
                "mean_degradation_percent": None if degradation_ratio is None else round(degradation_ratio * 100.0, 4),
                "min_ap": None if min_ap is None else round(min_ap, 4),
                "max_ap": None if max_ap is None else round(max_ap, 4),
                "worst_attack": worst_attack_name,
                "worst_attack_ap": round(worst_attack_ap, 4) if worst_attack_ap is not None else None,
                "worst_attack_degradation_percent": None
                if worst_attack_degradation is None
                else round(worst_attack_degradation, 4),
                "attacks": sorted({str(cell.get("attack", "")) for cell in sev_cells}),
            }
        )

    return results


def compute_run_classes_breakdown(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute per-class detection performance and degradation under attack.

    Args:
        report: Serialized RunReport dictionary.

    Returns:
        List of per-class detection analytics dictionaries sorted by class name.
    """
    sample_results = list(report.get("sample_results", []))
    if not sample_results:
        return []

    class_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "ground_truth_count": 0,
            "clean_detected_count": 0,
            "attacked_detected_count": 0,
            "lost_objects_count": 0,
            "hallucinated_objects_count": 0,
            "unaffected_objects_count": 0,
        }
    )

    for sample in sample_results:
        object_evidences = sample.get("object_evidence", [])
        if not object_evidences:
            continue

        for ev in object_evidences:
            if not isinstance(ev.get("clean_detected"), bool) or not isinstance(ev.get("attacked_detected"), bool):
                continue
            label = str(ev.get("label") or ev.get("ground_truth_label") or ev.get("class_name") or "unknown")
            stats = class_stats[label]
            stats["ground_truth_count"] += 1
            clean_hit = ev["clean_detected"]
            attack_hit = ev["attacked_detected"]

            if clean_hit:
                stats["clean_detected_count"] += 1
            if attack_hit:
                stats["attacked_detected_count"] += 1

            if clean_hit and not attack_hit:
                stats["lost_objects_count"] += 1
            elif not clean_hit and attack_hit:
                stats["hallucinated_objects_count"] += 1
            elif clean_hit and attack_hit:
                stats["unaffected_objects_count"] += 1

    class_breakdowns: list[dict[str, Any]] = []
    for class_name, stats in sorted(class_stats.items()):
        gt_count = stats["ground_truth_count"]
        clean_count = stats["clean_detected_count"]
        attack_count = stats["attacked_detected_count"]

        clean_rate = clean_count / gt_count if gt_count > 0 else 0.0
        attack_rate = attack_count / gt_count if gt_count > 0 else 0.0
        drop_ratio = max(0.0, (clean_rate - attack_rate) / clean_rate) if clean_rate > 0.0 else 0.0

        class_breakdowns.append(
            {
                "class_name": class_name,
                "total_ground_truth_objects": gt_count,
                "clean_detected_count": clean_count,
                "attacked_detected_count": attack_count,
                "clean_detection_rate": round(clean_rate, 4),
                "attacked_detection_rate": round(attack_rate, 4),
                "detection_drop_ratio": round(drop_ratio, 6),
                "detection_drop_percent": round(drop_ratio * 100.0, 4),
                "lost_objects_count": stats["lost_objects_count"],
                "hallucinated_objects_count": stats["hallucinated_objects_count"],
                "unaffected_objects_count": stats["unaffected_objects_count"],
            }
        )

    return class_breakdowns


def compute_run_samples_breakdown(
    report: Mapping[str, Any],
    *,
    attack: str | None = None,
    severity: int | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Extract granular sample-level evidence and object-level failure transitions.

    Args:
        report: Serialized RunReport dictionary.
        attack: Optional attack name filter.
        severity: Optional severity level filter.
        limit: Optional maximum number of items to return.
        offset: Offset for pagination.

    Returns:
        Dictionary containing total sample counts and paginated sample diagnostics.
    """
    sample_results = list(report.get("sample_results", []))

    filtered_samples: list[dict[str, Any]] = []
    for sample in sample_results:
        if attack is not None and sample.get("attack") != attack:
            continue
        if severity is not None and int(sample.get("severity", 0)) != severity:
            continue

        clean_pred = sample.get("clean_prediction") or {}
        attacked_pred = sample.get("attacked_prediction") or {}
        gt = sample.get("ground_truth") or {}

        clean_boxes = len(clean_pred.get("boxes", []))
        attacked_boxes = len(attacked_pred.get("boxes", []))
        gt_boxes = len(gt.get("boxes", []))

        transitions: list[str] = []
        for ev in sample.get("object_evidence", []):
            transition_str = ev.get("transition")
            if transition_str:
                transitions.append(str(transition_str))

        filtered_samples.append(
            {
                "sample_id": str(sample.get("sample_id", "")),
                "attack": str(sample.get("attack", "")),
                "severity": int(sample.get("severity", 1)),
                "degradation_hint": round(float(sample.get("degradation_hint", 0.0)), 6),
                "clean_boxes_count": clean_boxes,
                "attacked_boxes_count": attacked_boxes,
                "ground_truth_boxes_count": gt_boxes,
                "lost_detections_count": max(0, clean_boxes - attacked_boxes),
                "new_false_positives_count": max(0, attacked_boxes - clean_boxes),
                "object_transitions": transitions,
                "clean_image_path": sample.get("clean_image_path"),
                "attacked_image_path": sample.get("attacked_image_path"),
            }
        )

    # Sort primarily by largest degradation hint
    filtered_samples.sort(key=lambda s: s["degradation_hint"], reverse=True)

    total_count = len(filtered_samples)
    paginated_items = filtered_samples[offset : offset + limit] if limit is not None else filtered_samples[offset:]

    return {
        "total_samples": len(sample_results),
        "filtered_samples_count": total_count,
        "offset": offset,
        "limit": limit,
        "items": paginated_items,
    }


def _aggregate_attack_metrics(
    attack_cells_map: Mapping[str, list[dict[str, Any]]],
    clean_ap: float | None,
    failure_counts: Mapping[str, int] | None = None,
    *,
    task_id: str = "detection2d",
    metric_key: str = "map50_95",
    metric_label: str = "mAP@50-95",
    allow_legacy_ap: bool = True,
) -> list[dict[str, Any]]:
    """Helper to aggregate per-attack metrics over its severity cells."""
    summaries: list[dict[str, Any]] = []

    for attack_name, cell_list in attack_cells_map.items():
        if not cell_list:
            continue

        group_name = str(cell_list[0].get("group", "A"))
        category_name = cell_list[0].get("category")
        severities_evaluated = sorted(int(cell.get("severity", 1)) for cell in cell_list)

        ap_by_sev: dict[int, float] = {}
        deg_percent_by_sev: dict[int, float] = {}
        deg_ratio_by_sev: dict[int, float] = {}

        for cell in cell_list:
            sev = int(cell.get("severity", 1))
            cell_ap = _cell_primary_value(cell, task_id, metric_key, allow_legacy_ap=allow_legacy_ap)
            if cell_ap is None:
                continue
            ratio = max(0.0, (clean_ap - cell_ap) / clean_ap) if clean_ap is not None and clean_ap > 0.0 else None
            ap_by_sev[sev] = round(cell_ap, 4)
            if ratio is not None:
                deg_ratio_by_sev[sev] = round(ratio, 6)
                deg_percent_by_sev[sev] = round(ratio * 100.0, 4)

        aps = list(ap_by_sev.values())
        mean_ap = sum(aps) / len(aps) if aps else None
        min_ap = min(aps) if aps else None
        max_ap = max(aps) if aps else None

        mean_deg_ratio = (
            max(0.0, (clean_ap - mean_ap) / clean_ap)
            if clean_ap is not None and clean_ap > 0.0 and mean_ap is not None
            else None
        )
        worst_sev = min(ap_by_sev.items(), key=lambda item: item[1])[0] if ap_by_sev else 1

        total_samples = sum(int(cell.get("n_samples", 0)) for cell in cell_list)
        total_seconds = sum(float(cell.get("seconds", 0.0)) for cell in cell_list)

        fail_count = (failure_counts or {}).get(attack_name, 0)

        summaries.append(
            {
                "attack": attack_name,
                "group": group_name,
                "category": category_name,
                "metric_key": metric_key,
                "metric_label": metric_label,
                "severities_evaluated": severities_evaluated,
                "ap_by_severity": ap_by_sev,
                "degradation_percent_by_severity": deg_percent_by_sev,
                "degradation_ratio_by_severity": deg_ratio_by_sev,
                "mean_ap": None if mean_ap is None else round(mean_ap, 4),
                "mean_metric": None if mean_ap is None else round(mean_ap, 4),
                "min_ap": None if min_ap is None else round(min_ap, 4),
                "max_ap": None if max_ap is None else round(max_ap, 4),
                "mean_degradation_ratio": None if mean_deg_ratio is None else round(mean_deg_ratio, 6),
                "mean_degradation_percent": None if mean_deg_ratio is None else round(mean_deg_ratio * 100.0, 4),
                "worst_severity": worst_sev,
                "total_samples": total_samples,
                "total_seconds": round(total_seconds, 3),
                "failure_sample_count": fail_count,
            }
        )

    return summaries


def compute_run_distance_breakdown(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compute 3D perception performance breakdown by distance buckets (near, medium, far)."""
    clean_metrics = dict((report.get("metrics") or {}).get("clean") or {})
    cells = list(report.get("cells", []))

    clean_near = _finite_number(clean_metrics.get("kitti_3d_ap_near"))
    clean_medium = _finite_number(clean_metrics.get("kitti_3d_ap_medium"))
    clean_far = _finite_number(clean_metrics.get("kitti_3d_ap_far"))

    # Bucket failures from worst_cases
    worst_cases = list(report.get("worst_cases", []))
    near_failures = sum(1 for f in worst_cases if (f.get("metadata") or {}).get("distance_bucket") == "near")
    medium_failures = sum(1 for f in worst_cases if (f.get("metadata") or {}).get("distance_bucket") == "medium")
    far_failures = sum(1 for f in worst_cases if (f.get("metadata") or {}).get("distance_bucket") == "far")

    # Aggregate attack performance across cells
    attack_near_scores = []
    attack_medium_scores = []
    attack_far_scores = []

    for cell in cells:
        cell_metrics = cell.get("metrics") or {}
        if "kitti_3d_ap_near" in cell_metrics:
            attack_near_scores.append(float(cell_metrics["kitti_3d_ap_near"]))
        if "kitti_3d_ap_medium" in cell_metrics:
            attack_medium_scores.append(float(cell_metrics["kitti_3d_ap_medium"]))
        if "kitti_3d_ap_far" in cell_metrics:
            attack_far_scores.append(float(cell_metrics["kitti_3d_ap_far"]))

    mean_near_ap = sum(attack_near_scores) / len(attack_near_scores) if attack_near_scores else None
    mean_medium_ap = sum(attack_medium_scores) / len(attack_medium_scores) if attack_medium_scores else None
    mean_far_ap = sum(attack_far_scores) / len(attack_far_scores) if attack_far_scores else None

    def degradation(clean: float | None, attacked: float | None) -> float | None:
        if clean is None or attacked is None or clean <= 0.0:
            return None
        return max(0.0, (clean - attacked) / clean)

    deg_near = degradation(clean_near, mean_near_ap)
    deg_medium = degradation(clean_medium, mean_medium_ap)
    deg_far = degradation(clean_far, mean_far_ap)
    degradations = {"near": deg_near, "medium": deg_medium, "far": deg_far}
    measured_degradations = {name: value for name, value in degradations.items() if value is not None}

    return {
        "run_id": report.get("run_id", ""),
        "task": report.get("task", "detection3d"),
        "buckets": {
            "near": {
                "range_meters": "0-20m",
                "clean_ap": None if clean_near is None else round(clean_near, 4),
                "attacked_ap": None if mean_near_ap is None else round(mean_near_ap, 4),
                "degradation_percent": None if deg_near is None else round(deg_near * 100.0, 2),
                "failure_count": near_failures,
            },
            "medium": {
                "range_meters": "20-40m",
                "clean_ap": None if clean_medium is None else round(clean_medium, 4),
                "attacked_ap": None if mean_medium_ap is None else round(mean_medium_ap, 4),
                "degradation_percent": None if deg_medium is None else round(deg_medium * 100.0, 2),
                "failure_count": medium_failures,
            },
            "far": {
                "range_meters": "40m+",
                "clean_ap": None if clean_far is None else round(clean_far, 4),
                "attacked_ap": None if mean_far_ap is None else round(mean_far_ap, 4),
                "degradation_percent": None if deg_far is None else round(deg_far * 100.0, 2),
                "failure_count": far_failures,
            },
        },
        "most_vulnerable_distance": max(measured_degradations, key=measured_degradations.get)
        if measured_degradations
        else None,
        "data_state": "MEASURED" if measured_degradations else "NO_DATA",
    }
