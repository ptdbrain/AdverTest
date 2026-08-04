"""Detection metrics: IoU, greedy matching, average precision.

Single-IoU-threshold AP (AP50 by default), macro-averaged over the classes that
actually occur in the ground truth. Deliberately dependency-free so the loop
runs in CI; swap in ``pycocotools`` for the official AP@[.50:.95] before
reporting benchmark numbers (plan §3 metric 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.core.types import Box, Prediction, Sample

DEFAULT_IOU_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class DetectionSummary:
    """Class-aware TP/FP/FN counts at one IoU threshold."""

    true_positives: int
    false_positives: int
    false_negatives: int
    detections: int
    ground_truths: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "detections": self.detections,
            "ground_truths": self.ground_truths,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
        }


@dataclass(frozen=True, slots=True)
class AttackSuccessSummary:
    """Ground truths detected cleanly but lost after the attack."""

    clean_detected_truths: int
    lost_truths: int

    @property
    def rate(self) -> float:
        if not self.clean_detected_truths:
            return 0.0
        return self.lost_truths / self.clean_detected_truths

    def as_dict(self) -> dict[str, Any]:
        return {
            "clean_detected_truths": self.clean_detected_truths,
            "lost_truths": self.lost_truths,
            "rate": round(self.rate, 6),
        }


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


def detection_metric_suite(
    predictions: Sequence[Prediction],
    samples: Sequence[Sample],
) -> dict[str, Any]:
    """Detection metrics with explicit non-COCO provenance.

    The calculation uses the same deterministic, dependency-free matcher as
    the runner. It exposes AP50, AP75 and the ten-threshold mAP range for
    internal comparisons. ``metric_implementation`` makes it impossible for a
    caller to accidentally present it as the official pycocotools result.
    """
    thresholds = tuple(round(0.50 + index * 0.05, 2) for index in range(10))
    by_threshold = {threshold: average_precision(predictions, samples, threshold) for threshold in thresholds}
    per_class = average_precision_per_class(predictions, samples, 0.5)
    size_buckets = {"small": [], "medium": [], "large": []}
    for sample in samples:
        for box in sample.boxes:
            bucket = "small" if box.area < 32**2 else "medium" if box.area < 96**2 else "large"
            size_buckets[bucket].append(box)
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    by_size: dict[str, float | None] = {}
    for bucket, boxes in size_buckets.items():
        if not boxes:
            by_size[bucket] = None
            continue
        scoped_samples = [
            Sample(
                sample_id=sample.sample_id,
                image=sample.image,
                boxes=tuple(
                    box
                    for box in sample.boxes
                    if ("small" if box.area < 32**2 else "medium" if box.area < 96**2 else "large") == bucket
                ),
            )
            for sample in samples
        ]
        by_size[bucket] = average_precision(
            [by_id.get(sample.sample_id, Prediction(sample.sample_id)) for sample in scoped_samples],
            scoped_samples,
            0.5,
        )
    return {
        "metric_implementation": "advertest-greedy-interpolated-v1",
        "ap50": round(by_threshold[0.5], 6),
        "ap75": round(by_threshold[0.75], 6),
        "map50_95": round(float(np.mean(list(by_threshold.values()))), 6),
        "ap_per_class": {name: round(value, 6) for name, value in per_class.items()},
        "ap50_by_size": {name: None if value is None else round(value, 6) for name, value in by_size.items()},
    }


def bootstrap_average_precision(
    predictions: Sequence[Prediction],
    samples: Sequence[Sample],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    repetitions: int = 1000,
    seed: int = 20260730,
) -> tuple[float, float]:
    """Resample sample IDs and recompute AP for an empirical 95% interval."""
    if not samples:
        return (0.0, 0.0)
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(samples), len(samples))
        sampled = [samples[int(index)] for index in indices]
        # Duplicate source IDs need unique IDs because metric lookups are keyed
        # by sample ID. Preserve each paired prediction under a sampled alias.
        copied_samples: list[Sample] = []
        copied_predictions: list[Prediction] = []
        for position, sample in enumerate(sampled):
            alias = f"{sample.sample_id}#bootstrap-{position}"
            copied_samples.append(Sample(sample_id=alias, image=sample.image, boxes=sample.boxes))
            prediction = by_id.get(sample.sample_id, Prediction(sample.sample_id))
            copied_predictions.append(Prediction(alias, prediction.boxes, prediction.boxes3d, prediction.latency_ms))
        values.append(average_precision(copied_predictions, copied_samples, iou_threshold))
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def detection_summary(
    predictions: Sequence[Prediction],
    samples: Sequence[Sample],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> DetectionSummary:
    """Aggregate class-aware TP/FP/FN counts for a dataset."""
    by_sample = {prediction.sample_id: prediction for prediction in predictions}
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    detections = 0
    ground_truths = 0
    for sample in samples:
        predicted = by_sample.get(sample.sample_id, Prediction(sample.sample_id)).boxes
        matches = match_boxes(predicted, sample.boxes, iou_threshold)
        matched = len(matches)
        true_positives += matched
        false_positives += len(predicted) - matched
        false_negatives += len(sample.boxes) - matched
        detections += len(predicted)
        ground_truths += len(sample.boxes)
    return DetectionSummary(
        true_positives,
        false_positives,
        false_negatives,
        detections,
        ground_truths,
    )


def detection_attack_success_rate(
    clean_predictions: Sequence[Prediction],
    attacked_predictions: Sequence[Prediction],
    samples: Sequence[Sample],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> AttackSuccessSummary:
    """Rate of cleanly detected ground truths no longer detected after attack."""
    clean_by_sample = {prediction.sample_id: prediction for prediction in clean_predictions}
    attacked_by_sample = {prediction.sample_id: prediction for prediction in attacked_predictions}
    clean_detected = 0
    lost = 0
    for sample in samples:
        clean_boxes = clean_by_sample.get(
            sample.sample_id,
            Prediction(sample.sample_id),
        ).boxes
        attacked_boxes = attacked_by_sample.get(
            sample.sample_id,
            Prediction(sample.sample_id),
        ).boxes
        clean_truths = set(match_boxes(clean_boxes, sample.boxes, iou_threshold).values())
        attacked_truths = set(match_boxes(attacked_boxes, sample.boxes, iou_threshold).values())
        clean_detected += len(clean_truths)
        lost += len(clean_truths - attacked_truths)
    return AttackSuccessSummary(clean_detected, lost)


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
