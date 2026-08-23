"""Pure analytics aggregation functions for individual test run reports.

All functions in this module operate on dictionary representations of RunReport
(as produced by RunReport.as_dict() or retrieved from the database store).
No filesystem I/O or external network queries are performed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any


def compute_run_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """Compute high-level executive analytics for a single benchmark run.

    Args:
        report: Serialized RunReport dictionary.

    Returns:
        Dictionary containing overall performance, group-level breakdown,
        worst/resilient attack indicators, and robust score metrics.
    """
    clean_ap = float(report.get("ap_clean", 0.0))
    cells = list(report.get("cells", []))
    skipped = list(report.get("skipped", []))
    metrics = dict(report.get("metrics", {}))

    if not cells:
        return {
            "run_id": report.get("run_id", ""),
            "model": report.get("model", ""),
            "model_version": report.get("model_version", ""),
            "dataset": report.get("dataset", ""),
            "n_samples": int(report.get("n_samples", 0)),
            "ap_clean": round(clean_ap, 4),
            "mean_attack_ap": round(clean_ap, 4),
            "overall_degradation_ratio": 0.0,
            "overall_degradation_percent": 0.0,
            "robust_score": 100.0 if clean_ap > 0.0 else 0.0,
            "worst_attack": None,
            "most_resilient_attack": None,
            "group_summaries": {},
            "total_cells_evaluated": 0,
            "total_attacks_evaluated": 0,
            "total_skipped_attacks": len(skipped),
            "clean_metrics": metrics.get("clean", {}),
            "seconds": float(report.get("seconds", 0.0)),
        }

    attack_cells_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    group_cells_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for cell in cells:
        attack_name = str(cell.get("attack", ""))
        group_name = str(cell.get("group", "A"))
        attack_cells_map[attack_name].append(cell)
        group_cells_map[group_name].append(cell)

    mean_attack_ap = sum(float(cell.get("ap", 0.0)) for cell in cells) / len(cells)
    overall_degradation_ratio = (
        max(0.0, (clean_ap - mean_attack_ap) / clean_ap) if clean_ap > 0.0 else 0.0
    )

    attack_summaries = _aggregate_attack_metrics(attack_cells_map, clean_ap)
    sorted_by_degradation = sorted(
        attack_summaries, key=lambda item: item["mean_degradation_ratio"], reverse=True
    )

    worst_attack = sorted_by_degradation[0] if sorted_by_degradation else None
    most_resilient_attack = sorted_by_degradation[-1] if sorted_by_degradation else None

    group_summaries: dict[str, dict[str, Any]] = {}
    for group_name, group_cells in sorted(group_cells_map.items()):
        group_mean_ap = sum(float(cell.get("ap", 0.0)) for cell in group_cells) / len(group_cells)
        group_degradation_ratio = (
            max(0.0, (clean_ap - group_mean_ap) / clean_ap) if clean_ap > 0.0 else 0.0
        )
        group_attacks = sorted({str(cell.get("attack", "")) for cell in group_cells})
        group_summaries[group_name] = {
            "cell_count": len(group_cells),
            "mean_ap": round(group_mean_ap, 4),
            "mean_degradation_ratio": round(group_degradation_ratio, 6),
            "mean_degradation_percent": round(group_degradation_ratio * 100.0, 4),
            "attacks": group_attacks,
        }

    robust_score = max(0.0, 100.0 * (1.0 - overall_degradation_ratio))

    return {
        "run_id": report.get("run_id", ""),
        "model": report.get("model", ""),
        "model_version": report.get("model_version", ""),
        "dataset": report.get("dataset", ""),
        "n_samples": int(report.get("n_samples", 0)),
        "ap_clean": round(clean_ap, 4),
        "mean_attack_ap": round(mean_attack_ap, 4),
        "overall_degradation_ratio": round(overall_degradation_ratio, 6),
        "overall_degradation_percent": round(overall_degradation_ratio * 100.0, 4),
        "robust_score": round(robust_score, 2),
        "worst_attack": worst_attack,
        "most_resilient_attack": most_resilient_attack,
        "group_summaries": group_summaries,
        "total_cells_evaluated": len(cells),
        "total_attacks_evaluated": len(attack_cells_map),
        "total_skipped_attacks": len(skipped),
        "clean_metrics": metrics.get("clean", {}),
        "seconds": float(report.get("seconds", 0.0)),
    }


def compute_run_attacks_breakdown(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute detailed per-attack robustness analytics across all evaluated severities.

    Args:
        report: Serialized RunReport dictionary.

    Returns:
        List of per-attack breakdown dictionaries ranked from most degraded to least degraded.
    """
    clean_ap = float(report.get("ap_clean", 0.0))
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

    breakdowns = _aggregate_attack_metrics(
        attack_cells_map, clean_ap, failure_counts=attack_failure_counts
    )
    return sorted(breakdowns, key=lambda item: item["mean_degradation_ratio"], reverse=True)


def compute_run_severity_breakdown(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compute robustness analytics broken down by corruption/perturbation severity level (1..5).

    Args:
        report: Serialized RunReport dictionary.

    Returns:
        List of per-severity summary dictionaries sorted in ascending order of severity level.
    """
    clean_ap = float(report.get("ap_clean", 0.0))
    cells = list(report.get("cells", []))

    if not cells:
        return []

    severity_cells_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        sev = int(cell.get("severity", 1))
        severity_cells_map[sev].append(cell)

    results: list[dict[str, Any]] = []
    for severity_level, sev_cells in sorted(severity_cells_map.items()):
        aps = [float(cell.get("ap", 0.0)) for cell in sev_cells]
        mean_ap = sum(aps) / len(aps) if aps else 0.0
        min_ap = min(aps) if aps else 0.0
        max_ap = max(aps) if aps else 0.0
        degradation_ratio = (
            max(0.0, (clean_ap - mean_ap) / clean_ap) if clean_ap > 0.0 else 0.0
        )

        worst_cell = min(sev_cells, key=lambda c: float(c.get("ap", 0.0))) if sev_cells else None
        worst_attack_name = str(worst_cell.get("attack", "")) if worst_cell else None
        worst_attack_ap = float(worst_cell.get("ap", 0.0)) if worst_cell else None
        worst_attack_degradation = (
            max(0.0, (clean_ap - worst_attack_ap) / clean_ap * 100.0)
            if clean_ap > 0.0 and worst_attack_ap is not None
            else 0.0
        )

        results.append({
            "severity": severity_level,
            "evaluated_cells_count": len(sev_cells),
            "mean_ap": round(mean_ap, 4),
            "mean_degradation_ratio": round(degradation_ratio, 6),
            "mean_degradation_percent": round(degradation_ratio * 100.0, 4),
            "min_ap": round(min_ap, 4),
            "max_ap": round(max_ap, 4),
            "worst_attack": worst_attack_name,
            "worst_attack_ap": round(worst_attack_ap, 4) if worst_attack_ap is not None else None,
            "worst_attack_degradation_percent": round(worst_attack_degradation, 4),
            "attacks": sorted({str(cell.get("attack", "")) for cell in sev_cells}),
        })

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

    class_stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "ground_truth_count": 0,
        "clean_detected_count": 0,
        "attacked_detected_count": 0,
        "lost_objects_count": 0,
        "hallucinated_objects_count": 0,
        "unaffected_objects_count": 0,
    })

    for sample in sample_results:
        object_evidences = sample.get("object_evidence", [])
        if not object_evidences:
            _extract_classes_from_predictions(sample, class_stats)
            continue

        for ev in object_evidences:
            label = str(ev.get("label") or ev.get("ground_truth_label") or ev.get("class_name") or "unknown")
            stats = class_stats[label]
            stats["ground_truth_count"] += 1
            clean_hit = bool(ev.get("clean_detected", True))
            attack_hit = bool(ev.get("attacked_detected", False))

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

        class_breakdowns.append({
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
        })

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

        filtered_samples.append({
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
        })

    # Sort primarily by largest degradation hint
    filtered_samples.sort(key=lambda s: s["degradation_hint"], reverse=True)

    total_count = len(filtered_samples)
    paginated_items = (
        filtered_samples[offset : offset + limit] if limit is not None else filtered_samples[offset:]
    )

    return {
        "total_samples": len(sample_results),
        "filtered_samples_count": total_count,
        "offset": offset,
        "limit": limit,
        "items": paginated_items,
    }


def _aggregate_attack_metrics(
    attack_cells_map: Mapping[str, list[dict[str, Any]]],
    clean_ap: float,
    failure_counts: Mapping[str, int] | None = None,
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
            cell_ap = float(cell.get("ap", 0.0))
            ratio = max(0.0, (clean_ap - cell_ap) / clean_ap) if clean_ap > 0.0 else 0.0
            ap_by_sev[sev] = round(cell_ap, 4)
            deg_ratio_by_sev[sev] = round(ratio, 6)
            deg_percent_by_sev[sev] = round(ratio * 100.0, 4)

        aps = list(ap_by_sev.values())
        mean_ap = sum(aps) / len(aps) if aps else 0.0
        min_ap = min(aps) if aps else 0.0
        max_ap = max(aps) if aps else 0.0

        mean_deg_ratio = max(0.0, (clean_ap - mean_ap) / clean_ap) if clean_ap > 0.0 else 0.0
        worst_sev = min(ap_by_sev.items(), key=lambda item: item[1])[0] if ap_by_sev else 1

        total_samples = sum(int(cell.get("n_samples", 0)) for cell in cell_list)
        total_seconds = sum(float(cell.get("seconds", 0.0)) for cell in cell_list)

        fail_count = (failure_counts or {}).get(attack_name, 0)

        summaries.append({
            "attack": attack_name,
            "group": group_name,
            "category": category_name,
            "severities_evaluated": severities_evaluated,
            "ap_by_severity": ap_by_sev,
            "degradation_percent_by_severity": deg_percent_by_sev,
            "degradation_ratio_by_severity": deg_ratio_by_sev,
            "mean_ap": round(mean_ap, 4),
            "min_ap": round(min_ap, 4),
            "max_ap": round(max_ap, 4),
            "mean_degradation_ratio": round(mean_deg_ratio, 6),
            "mean_degradation_percent": round(mean_deg_ratio * 100.0, 4),
            "worst_severity": worst_sev,
            "total_samples": total_samples,
            "total_seconds": round(total_seconds, 3),
            "failure_sample_count": fail_count,
        })

    return summaries


def _extract_classes_from_predictions(
    sample: Mapping[str, Any], class_stats: dict[str, dict[str, int]]
) -> None:
    """Fallback class extraction when explicit object_evidence is missing."""
    clean_pred = sample.get("clean_prediction") or {}
    attacked_pred = sample.get("attacked_prediction") or {}
    gt = sample.get("ground_truth") or {}

    labels = set(gt.get("labels", [])) | set(clean_pred.get("labels", [])) | set(attacked_pred.get("labels", []))
    for label in labels:
        label_str = str(label)
        clean_hits = clean_pred.get("labels", []).count(label)
        attack_hits = attacked_pred.get("labels", []).count(label)
        gt_hits = gt.get("labels", []).count(label)

        stats = class_stats[label_str]
        stats["ground_truth_count"] += gt_hits or max(clean_hits, 1)
        stats["clean_detected_count"] += clean_hits
        stats["attacked_detected_count"] += attack_hits
        stats["lost_objects_count"] += max(0, clean_hits - attack_hits)
        stats["hallucinated_objects_count"] += max(0, attack_hits - clean_hits)
        stats["unaffected_objects_count"] += min(clean_hits, attack_hits)
