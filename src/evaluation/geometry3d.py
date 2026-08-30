"""Oriented 3D Bounding Box Geometry, IoU calculation, and AP evaluation.

Implements:
- Oriented polygon intersection on Bird's-Eye-View (BEV).
- Height/Z-axis overlap calculation.
- Full 3D Oriented Bounding Box IoU: IoU_3D = Inter_Vol / (Vol_A + Vol_B - Inter_Vol).
- Robust edge-case handling: rotated boxes, zero/negative dims, NaN/Inf, touching edges.
- KITTI class-specific 3D evaluation: Car @ 0.7, Pedestrian @ 0.5, Cyclist @ 0.5.
- Greedy score-ordered 1-to-1 bipartite matching.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from src.core.types import Box3D

try:
    from shapely.geometry import Polygon

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

KITTI_3D_IOU_THRESHOLDS: dict[str, float] = {
    "Car": 0.7,
    "car": 0.7,
    "Pedestrian": 0.5,
    "pedestrian": 0.5,
    "Cyclist": 0.5,
    "cyclist": 0.5,
}


@dataclass(frozen=True, slots=True)
class Match3D:
    gt_index: int
    prediction_index: int
    iou: float


def get_kitti_threshold(label: str, default: float = 0.5) -> float:
    """Return KITTI 3D benchmark threshold for a given class name."""
    return KITTI_3D_IOU_THRESHOLDS.get(label, default)


def _is_valid_box(b: Box3D) -> bool:
    """Validate that box dimensions are positive and finite numbers."""
    values = (b.x, b.y, b.z, b.length, b.width, b.height, b.yaw)
    for v in values:
        if not math.isfinite(v):
            return False
    return b.length > 0 and b.width > 0 and b.height > 0


def _get_bev_polygon(box: Box3D) -> Polygon:
    l2 = box.length / 2.0
    w2 = box.width / 2.0
    corners = [(-l2, -w2), (l2, -w2), (l2, w2), (-l2, w2)]
    cos_a = math.cos(box.yaw)
    sin_a = math.sin(box.yaw)
    rotated_corners = []
    for cx, cy in corners:
        rx = cx * cos_a - cy * sin_a + box.x
        ry = cx * sin_a + cy * cos_a + box.y
        rotated_corners.append((rx, ry))
    return Polygon(rotated_corners)


def _bev_intersection_area(a: Box3D, b: Box3D) -> float:
    """Compute 2D intersection area of rotated bounding boxes on BEV plane."""
    if not _is_valid_box(a) or not _is_valid_box(b):
        return 0.0

    if HAS_SHAPELY:
        poly_a = _get_bev_polygon(a)
        poly_b = _get_bev_polygon(b)
        if not poly_a.intersects(poly_b):
            return 0.0
        inter = poly_a.intersection(poly_b).area
        return float(max(0.0, inter))
    else:
        # Axis-aligned approximation fallback
        ax1, ay1 = a.x - a.length / 2, a.y - a.width / 2
        ax2, ay2 = a.x + a.length / 2, a.y + a.width / 2
        bx1, by1 = b.x - b.length / 2, b.y - b.width / 2
        bx2, by2 = b.x + b.length / 2, b.y + b.width / 2

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        return float(iw * ih)


def bev_iou(a: Box3D, b: Box3D) -> float:
    """Bird's-eye-view 2D IoU of two rotated 3D boxes (ignores Z-axis)."""
    if not _is_valid_box(a) or not _is_valid_box(b):
        return 0.0

    inter_area = _bev_intersection_area(a, b)
    if inter_area <= 0.0:
        return 0.0

    area_a = a.length * a.width
    area_b = b.length * b.width
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, inter_area / union_area)))


def box3d_iou(a: Box3D, b: Box3D) -> float:
    """Full 3D Oriented Bounding Box IoU: volume_inter / volume_union."""
    if not _is_valid_box(a) or not _is_valid_box(b):
        return 0.0

    # 1. 2D BEV Intersection Area
    inter_area_2d = _bev_intersection_area(a, b)
    if inter_area_2d <= 0.0:
        return 0.0

    # 2. Z-axis Overlap
    az_min, az_max = a.z - a.height / 2.0, a.z + a.height / 2.0
    bz_min, bz_max = b.z - b.height / 2.0, b.z + b.height / 2.0
    z_inter_min = max(az_min, bz_min)
    z_inter_max = min(az_max, bz_max)
    z_overlap = max(0.0, z_inter_max - z_inter_min)

    if z_overlap <= 0.0:
        return 0.0

    # 3. 3D Volumes
    inter_volume = inter_area_2d * z_overlap
    vol_a = a.length * a.width * a.height
    vol_b = b.length * b.width * b.height
    union_volume = vol_a + vol_b - inter_volume

    if union_volume <= 0.0:
        return 0.0

    return float(min(1.0, max(0.0, inter_volume / union_volume)))


def match_boxes3d(
    ground_truth: tuple[Box3D, ...],
    predictions: tuple[Box3D, ...],
    *,
    iou_threshold: float | None = None,
    use_3d_iou: bool = True,
) -> tuple[Match3D, ...]:
    """Greedy score-ordered matching. One GT matches at most one prediction."""
    pred_indices = sorted(range(len(predictions)), key=lambda i: predictions[i].score or 0.0, reverse=True)

    matches: list[Match3D] = []
    matched_gt: set[int] = set()

    for p_idx in pred_indices:
        pred = predictions[p_idx]
        best_gt_idx = -1
        best_iou = -1.0

        # Class-specific threshold if not explicitly passed
        threshold = iou_threshold if iou_threshold is not None else get_kitti_threshold(pred.label, 0.5)

        for g_idx, gt in enumerate(ground_truth):
            if g_idx in matched_gt:
                continue
            if gt.label != pred.label:
                continue

            iou = box3d_iou(gt, pred) if use_3d_iou else bev_iou(gt, pred)
            if iou >= threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            matches.append(Match3D(gt_index=best_gt_idx, prediction_index=p_idx, iou=best_iou))

    return tuple(matches)


def evaluate_3d_detections_ap(
    ground_truths: list[tuple[Box3D, ...]],
    predictions_list: list[tuple[Box3D, ...]],
    use_3d_iou: bool = True,
) -> dict[str, float]:
    """Compute per-class AP and overall 3D mAP according to KITTI standard."""
    tp_scores: dict[str, list[float]] = defaultdict(list)
    fp_scores: dict[str, list[float]] = defaultdict(list)
    gt_counts: dict[str, int] = defaultdict(int)

    for gt_boxes, pred_boxes in zip(ground_truths, predictions_list):
        for gt in gt_boxes:
            gt_counts[gt.label] += 1

        matches = match_boxes3d(gt_boxes, pred_boxes, use_3d_iou=use_3d_iou)
        matched_pred_indices = {m.prediction_index: m.iou for m in matches}

        for idx, pred in enumerate(pred_boxes):
            score = pred.score if pred.score is not None else 1.0
            if idx in matched_pred_indices:
                tp_scores[pred.label].append(score)
            else:
                fp_scores[pred.label].append(score)

    per_class_ap: dict[str, float] = {}
    for label, n_gt in gt_counts.items():
        if n_gt == 0:
            per_class_ap[label] = 0.0
            continue

        scores_tp = tp_scores.get(label, [])
        scores_fp = fp_scores.get(label, [])
        all_detections = [(s, 1) for s in scores_tp] + [(s, 0) for s in scores_fp]
        all_detections.sort(key=lambda x: x[0], reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []

        for _, is_tp in all_detections:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
            recalls.append(tp_cumsum / n_gt)

        # 11-point interpolated AP (KITTI metric standard)
        ap = 0.0
        for r_thresh in [i / 10.0 for i in range(11)]:
            prec_at_r = [p for p, r in zip(precisions, recalls) if r >= r_thresh]
            ap += (max(prec_at_r) if prec_at_r else 0.0) / 11.0
        per_class_ap[label] = round(ap, 4)

    if per_class_ap:
        per_class_ap["mAP_3D"] = round(sum(per_class_ap.values()) / len(per_class_ap), 4)
    else:
        per_class_ap["mAP_3D"] = 0.0

    return per_class_ap
