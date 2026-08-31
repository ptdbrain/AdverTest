import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.core.types import Box3D

try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

@dataclass(frozen=True, slots=True)
class Match3D:
    gt_index: int
    prediction_index: int
    iou: float


_KITTI_IOU_THRESHOLDS = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}


def get_kitti_threshold(label: str) -> float:
    """Return the official KITTI 3D IoU threshold for a normalized class."""
    return _KITTI_IOU_THRESHOLDS.get(label, 0.5)


def _is_valid_box(box: Box3D) -> bool:
    values = (box.x, box.y, box.z, box.length, box.width, box.height, box.yaw)
    return all(math.isfinite(value) for value in values) and all(
        value > 0 for value in (box.length, box.width, box.height)
    )

def bev_iou(a: Box3D, b: Box3D) -> float:
    """Bird's-eye-view IoU of two rotated 3D boxes (ignores Z)."""
    if HAS_SHAPELY:
        poly_a = _get_bev_polygon(a)
        poly_b = _get_bev_polygon(b)
        if not poly_a.intersects(poly_b):
            return 0.0
        inter = poly_a.intersection(poly_b).area
        union = poly_a.area + poly_b.area - inter
        return float(inter / union) if union > 0 else 0.0
    intersection = _bev_intersection_area(a, b)
    union = a.length * a.width + b.length * b.width - intersection
    return float(intersection / union) if union > 0 else 0.0


def box3d_iou(a: Box3D, b: Box3D) -> float:
    """Volumetric IoU for oriented cuboids with robust invalid-input handling."""
    if not _is_valid_box(a) or not _is_valid_box(b):
        return 0.0

    if HAS_SHAPELY:
        poly_a = _get_bev_polygon(a)
        poly_b = _get_bev_polygon(b)
        if not poly_a.intersects(poly_b):
            return 0.0
        intersection_area = float(poly_a.intersection(poly_b).area)
    else:
        intersection_area = _bev_intersection_area(a, b)
    if intersection_area <= 0:
        return 0.0

    z_overlap = min(a.z + a.height / 2, b.z + b.height / 2) - max(a.z - a.height / 2, b.z - b.height / 2)
    intersection_volume = intersection_area * max(0.0, z_overlap)
    if intersection_volume <= 0:
        return 0.0
    volume_a = a.length * a.width * a.height
    volume_b = b.length * b.width * b.height
    union = volume_a + volume_b - intersection_volume
    return float(intersection_volume / union) if union > 0 else 0.0

def _get_bev_polygon(box: Box3D):
    return Polygon(_bev_corners(box))


def _bev_corners(box: Box3D) -> list[tuple[float, float]]:
    l2, w2 = box.length / 2, box.width / 2
    corners = [(-l2, -w2), (l2, -w2), (l2, w2), (-l2, w2)]
    cos_a = math.cos(box.yaw)
    sin_a = math.sin(box.yaw)
    rotated_corners = []
    for cx, cy in corners:
        rx = cx * cos_a - cy * sin_a + box.x
        ry = cx * sin_a + cy * cos_a + box.y
        rotated_corners.append((rx, ry))
    return rotated_corners


def _bev_intersection_area(a: Box3D, b: Box3D) -> float:
    """Exact convex-polygon clipping fallback when Shapely is unavailable."""
    subject = _bev_corners(a)
    clip = _bev_corners(b)
    for edge_start, edge_end in zip(clip, clip[1:] + clip[:1], strict=True):
        if not subject:
            return 0.0
        output: list[tuple[float, float]] = []
        previous = subject[-1]
        for current in subject:
            current_inside = _is_left_of_edge(current, edge_start, edge_end)
            previous_inside = _is_left_of_edge(previous, edge_start, edge_end)
            if current_inside != previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            if current_inside:
                output.append(current)
            previous = current
        subject = output
    return _polygon_area(subject)


def _is_left_of_edge(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> bool:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]) >= -1e-12


def _line_intersection(
    first_start: tuple[float, float], first_end: tuple[float, float], second_start: tuple[float, float], second_end: tuple[float, float]
) -> tuple[float, float]:
    first_dx, first_dy = first_end[0] - first_start[0], first_end[1] - first_start[1]
    second_dx, second_dy = second_end[0] - second_start[0], second_end[1] - second_start[1]
    denominator = first_dx * second_dy - first_dy * second_dx
    if abs(denominator) < 1e-12:
        return first_end
    scale = ((second_start[0] - first_start[0]) * second_dy - (second_start[1] - first_start[1]) * second_dx) / denominator
    return first_start[0] + scale * first_dx, first_start[1] + scale * first_dy


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    vertices = list(points)
    return abs(
        sum(
            first[0] * second[1] - first[1] * second[0]
            for first, second in zip(vertices, vertices[1:] + vertices[:1], strict=True)
        )
    ) / 2

def match_boxes3d(
    ground_truth: tuple[Box3D, ...],
    predictions: tuple[Box3D, ...],
    *,
    iou_threshold: float | None = None,
    use_3d_iou: bool = False,
) -> tuple[Match3D, ...]:
    """Greedy score-ordered BEV matching. One GT matches at most one prediction."""
    pred_indices = sorted(range(len(predictions)), key=lambda i: predictions[i].score or 0.0, reverse=True)

    matches = []
    matched_gt = set()

    for p_idx in pred_indices:
        pred = predictions[p_idx]
        best_gt_idx = -1
        best_iou = -1.0

        for g_idx, gt in enumerate(ground_truth):
            if g_idx in matched_gt:
                continue
            if gt.label != pred.label:
                continue

            threshold = iou_threshold if iou_threshold is not None else get_kitti_threshold(gt.label)
            iou = box3d_iou(gt, pred) if use_3d_iou else bev_iou(gt, pred)
            if iou >= threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            matches.append(Match3D(gt_index=best_gt_idx, prediction_index=p_idx, iou=best_iou))

    return tuple(matches)


def evaluate_3d_detections_ap(
    ground_truth_frames: Sequence[Sequence[Box3D]],
    prediction_frames: Sequence[Sequence[Box3D]],
    *,
    use_3d_iou: bool = True,
) -> dict[str, float]:
    """Compute score-ranked, per-class AP over aligned frames.

    This evaluator intentionally keeps the data contract small: each frame is
    represented by normalized ``Box3D`` objects and predictions are matched at
    the KITTI class threshold.  It never fabricates an AP for classes without
    ground truth.
    """
    if len(ground_truth_frames) != len(prediction_frames):
        raise ValueError("ground_truth_frames and prediction_frames must have equal length")
    labels = sorted({box.label for frame in ground_truth_frames for box in frame})
    results: dict[str, float] = {}
    for label in labels:
        total_ground_truth = sum(1 for frame in ground_truth_frames for box in frame if box.label == label)
        ranked = sorted(
            (
                (prediction.score, frame_index, prediction)
                for frame_index, frame in enumerate(prediction_frames)
                for prediction in frame
                if prediction.label == label
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        matched: set[tuple[int, int]] = set()
        true_positives: list[int] = []
        false_positives: list[int] = []
        for _, frame_index, prediction in ranked:
            candidates = [
                (index, box)
                for index, box in enumerate(ground_truth_frames[frame_index])
                if box.label == label and (frame_index, index) not in matched
            ]
            score = box3d_iou if use_3d_iou else bev_iou
            best_index, best_iou = -1, -1.0
            for index, ground_truth in candidates:
                overlap = score(ground_truth, prediction)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0 and best_iou >= get_kitti_threshold(label):
                matched.add((frame_index, best_index))
                true_positives.append(1)
                false_positives.append(0)
            else:
                true_positives.append(0)
                false_positives.append(1)
        cumulative_tp = 0
        cumulative_fp = 0
        precisions: list[float] = []
        recalls: list[float] = []
        for tp, fp in zip(true_positives, false_positives, strict=True):
            cumulative_tp += tp
            cumulative_fp += fp
            precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))
            recalls.append(cumulative_tp / total_ground_truth)
        ap = 0.0
        previous_recall = 0.0
        for precision, recall in zip(reversed(precisions), reversed(recalls), strict=True):
            ap += max(precision, 0.0) * max(0.0, recall - previous_recall)
            previous_recall = max(previous_recall, recall)
        results[label] = ap
    results["mAP_3D"] = sum(results.values()) / len(results) if results else 0.0
    return results
