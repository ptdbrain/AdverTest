"""Detection metrics: IoU, greedy matching, average precision.

Single-IoU-threshold AP (AP50 by default), macro-averaged over the classes that
actually occur in the ground truth. Deliberately dependency-free so the loop
runs in CI; swap in ``pycocotools`` for the official AP@[.50:.95] before
reporting benchmark numbers (plan §3 metric 1).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.types import Box, Prediction, Sample

DEFAULT_IOU_THRESHOLD = 0.5


def iou(first: Box, second: Box) -> float:
    """Intersection over union of two boxes; 0.0 when they do not overlap."""
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = first.area + second.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_boxes(
    predictions: Sequence[Box],
    truths: Sequence[Box],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[int, int]:
    """Greedy score-ordered matching; returns ``{prediction index: truth index}``.

    A truth box is used at most once, and only same-class pairs may match — the
    convention behind "correctly detected" in the ASR definitions of plan §3.
    """
    order = sorted(range(len(predictions)), key=lambda index: predictions[index].score, reverse=True)
    taken: set[int] = set()
    matches: dict[int, int] = {}
    for prediction_index in order:
        prediction = predictions[prediction_index]
        best_index, best_iou = -1, iou_threshold
        for truth_index, truth in enumerate(truths):
            if truth_index in taken or truth.label != prediction.label:
                continue
            overlap = iou(prediction, truth)
            if overlap >= best_iou:
                best_index, best_iou = truth_index, overlap
        if best_index >= 0:
            taken.add(best_index)
            matches[prediction_index] = best_index
    return matches


def average_precision(
    predictions: Sequence[Prediction],
    samples: Sequence[Sample],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Macro-averaged AP over classes present in the ground truth."""
    per_class = average_precision_per_class(predictions, samples, iou_threshold)
    return float(np.mean(list(per_class.values()))) if per_class else 0.0


def average_precision_per_class(
    predictions: Sequence[Prediction],
    samples: Sequence[Sample],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, float]:
    """AP for every class that appears in the ground truth."""
    by_sample = {prediction.sample_id: prediction for prediction in predictions}
    labels = {box.label for sample in samples for box in sample.boxes}
    scored: dict[str, float] = {}
    for label in sorted(labels):
        flags, n_truths = _collect_hits(label, samples, by_sample, iou_threshold)
        scored[label] = _average_precision_from_hits(flags, n_truths)
    return scored


def _collect_hits(
    label: str,
    samples: Sequence[Sample],
    by_sample: dict[str, Prediction],
    iou_threshold: float,
) -> tuple[list[tuple[float, bool]], int]:
    """Per-class ``(score, is_true_positive)`` pairs plus the ground-truth count."""
    flags: list[tuple[float, bool]] = []
    n_truths = 0
    for sample in samples:
        truths = [box for box in sample.boxes if box.label == label]
        n_truths += len(truths)
        candidates = [box for box in by_sample.get(sample.sample_id, Prediction(sample.sample_id)).boxes]
        selected = [box for box in candidates if box.label == label]
        matches = match_boxes(selected, truths, iou_threshold)
        flags.extend((box.score, index in matches) for index, box in enumerate(selected))
    return flags, n_truths


def _average_precision_from_hits(flags: list[tuple[float, bool]], n_truths: int) -> float:
    """All-point interpolated AP from score-ranked TP/FP flags."""
    if n_truths == 0:
        return 0.0
    if not flags:
        return 0.0
    flags.sort(key=lambda item: item[0], reverse=True)
    hits = np.array([1.0 if is_hit else 0.0 for _, is_hit in flags], dtype=np.float64)
    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(1.0 - hits)
    recall = true_positives / n_truths
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-12)
    # Make precision monotonically decreasing, then integrate over recall steps.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall_steps = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * recall_steps))
