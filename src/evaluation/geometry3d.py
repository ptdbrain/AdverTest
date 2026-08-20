import math
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
    else:
        # Axis-aligned approximation
        ax1, ay1 = a.x - a.length / 2, a.y - a.width / 2
        ax2, ay2 = a.x + a.length / 2, a.y + a.width / 2
        bx1, by1 = b.x - b.length / 2, b.y - b.width / 2
        bx2, by2 = b.x + b.length / 2, b.y + b.width / 2
        
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

def _get_bev_polygon(box: Box3D):
    l2 = box.length / 2
    w2 = box.width / 2
    corners = [
        (-l2, -w2),
        (l2, -w2),
        (l2, w2),
        (-l2, w2)
    ]
    cos_a = math.cos(box.yaw)
    sin_a = math.sin(box.yaw)
    rotated_corners = []
    for cx, cy in corners:
        rx = cx * cos_a - cy * sin_a + box.x
        ry = cx * sin_a + cy * cos_a + box.y
        rotated_corners.append((rx, ry))
    return Polygon(rotated_corners)

def match_boxes3d(
    ground_truth: tuple[Box3D, ...],
    predictions: tuple[Box3D, ...],
    *,
    iou_threshold: float = 0.5,
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
            
            iou = bev_iou(gt, pred)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
                
        if best_gt_idx != -1:
            matched_gt.add(best_gt_idx)
            matches.append(Match3D(gt_index=best_gt_idx, prediction_index=p_idx, iou=best_iou))
            
    return tuple(matches)
