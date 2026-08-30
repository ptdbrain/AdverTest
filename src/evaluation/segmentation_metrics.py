"""Segmentation metrics and versioned failure decisions for SAM2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from src.core.types import ModelPrediction, Sample, SegmentationPrediction
from src.evaluation.base import EvaluationResult
from src.evaluation.contracts import FailureCase, MetricEnvelope
from src.segmentation.protocol import validate_segmentation_sample

if TYPE_CHECKING:
    from src.pipeline.protocol import BenchmarkProtocol

METRIC_IMPLEMENTATION = "advertest-segmentation-v1"
FailureReason = Literal[
    "correct",
    "iou_below_threshold",
    "empty_prediction",
    "wrong_object",
    "split_severe",
    "merge_severe",
    "boundary_error",
]


@dataclass(frozen=True, slots=True)
class MaskFailurePolicy:
    version: str = "sam-mask-failure-v1"
    iou_threshold: float = 0.5
    boundary_iou_threshold: float = 0.5
    severe_component_count: int = 3


@dataclass(frozen=True, slots=True)
class MaskEvaluation:
    sample_id: str
    object_id: str
    iou: float
    dice: float
    pixel_precision: float
    pixel_recall: float
    boundary_iou: float
    boundary_f_score: float
    confidence: float
    predicted_area_ratio: float
    failure_reason: FailureReason
    object_size: str

    @property
    def failed(self) -> bool:
        return self.failure_reason != "correct"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "object_id": self.object_id,
            "iou": self.iou,
            "dice": self.dice,
            "pixel_precision": self.pixel_precision,
            "pixel_recall": self.pixel_recall,
            "boundary_iou": self.boundary_iou,
            "boundary_f_score": self.boundary_f_score,
            "confidence": self.confidence,
            "predicted_area_ratio": self.predicted_area_ratio,
            "failure_reason": self.failure_reason,
            "failed": self.failed,
            "object_size": self.object_size,
        }


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted, truth = _binary_pair(prediction, target)
    union = np.logical_or(predicted, truth).sum()
    return float(np.logical_and(predicted, truth).sum() / union) if union else 1.0


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    predicted, truth = _binary_pair(prediction, target)
    denominator = predicted.sum() + truth.sum()
    return float(2 * np.logical_and(predicted, truth).sum() / denominator) if denominator else 1.0


def pixel_precision_recall(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    predicted, truth = _binary_pair(prediction, target)
    hit = np.logical_and(predicted, truth).sum()
    precision = float(hit / predicted.sum()) if predicted.any() else 0.0
    recall = float(hit / truth.sum()) if truth.any() else 0.0
    return precision, recall


def boundary_iou(prediction: np.ndarray, target: np.ndarray, *, tolerance: int = 2) -> float:
    predicted, truth = _binary_pair(prediction, target)
    pred_band = _dilate(_boundary(predicted), tolerance)
    truth_band = _dilate(_boundary(truth), tolerance)
    union = np.logical_or(pred_band, truth_band).sum()
    return float(np.logical_and(pred_band, truth_band).sum() / union) if union else 1.0


def boundary_f_score(prediction: np.ndarray, target: np.ndarray, *, tolerance: int = 2) -> float:
    predicted, truth = _binary_pair(prediction, target)
    pred_boundary, truth_boundary = _boundary(predicted), _boundary(truth)
    if not pred_boundary.any() and not truth_boundary.any():
        return 1.0
    if not pred_boundary.any() or not truth_boundary.any():
        return 0.0
    precision = float(np.logical_and(pred_boundary, _dilate(truth_boundary, tolerance)).sum() / pred_boundary.sum())
    recall = float(np.logical_and(truth_boundary, _dilate(pred_boundary, tolerance)).sum() / truth_boundary.sum())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_prediction(
    sample: Sample,
    prediction: SegmentationPrediction,
    *,
    policy: MaskFailurePolicy = MaskFailurePolicy(),
) -> list[MaskEvaluation]:
    """Evaluate one prompt-aligned prediction against reviewed instance masks."""
    validation = validate_segmentation_sample(sample, role="locked_test")
    if prediction.sample_id != sample.sample_id:
        raise ValueError("prediction/sample IDs do not match")
    results: list[MaskEvaluation] = []
    if not prediction.prompt_id:
        raise ValueError("SAM benchmark prediction must retain its fixed prompt_id")
    for instance in prediction.instances:
        mask = instance.mask
        if mask.shape != sample.mask.shape:  # type: ignore[union-attr]
            raise ValueError("prediction mask must use original sample resolution")
        object_id = instance.instance_id
        target_id = int(object_id)
        if target_id not in validation.instance_ids:
            raise ValueError(f"prompt object ID {target_id} does not exist in ground truth")
        target = sample.mask == target_id  # type: ignore[operator]
        predicted = np.asarray(mask, dtype=bool)
        iou = binary_iou(predicted, target)
        dice = dice_score(predicted, target)
        precision, recall = pixel_precision_recall(predicted, target)
        boundary = boundary_iou(predicted, target)
        f_score = boundary_f_score(predicted, target)
        reason = _failure_reason(predicted, sample.mask, target_id, iou, boundary, policy)  # type: ignore[arg-type]
        results.append(
            MaskEvaluation(
                sample_id=sample.sample_id,
                object_id=object_id,
                iou=iou,
                dice=dice,
                pixel_precision=precision,
                pixel_recall=recall,
                boundary_iou=boundary,
                boundary_f_score=f_score,
                confidence=float(instance.score),
                predicted_area_ratio=float(predicted.sum() / target.sum()) if target.any() else 0.0,
                failure_reason=reason,
                object_size=validation.object_sizes[target_id],
            )
        )
    return results


def segmentation_metric_suite(
    predictions: Sequence[SegmentationPrediction],
    samples: Sequence[Sample],
    *,
    policy: MaskFailurePolicy = MaskFailurePolicy(),
) -> dict[str, Any]:
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    evaluations = [
        item
        for sample in samples
        if sample.sample_id in by_id
        for item in evaluate_prediction(sample, by_id[sample.sample_id], policy=policy)
    ]
    if not evaluations:
        return {
            "metric_implementation": METRIC_IMPLEMENTATION,
            "mask_count": 0,
            "miou": 0.0,
            "failure_rate_ratio": 0.0,
            "failure_rate_pct": 0.0,
        }

    def mean(field: str) -> float:
        return float(np.mean([getattr(item, field) for item in evaluations]))

    grouped: dict[str, list[MaskEvaluation]] = defaultdict(list)
    by_class: dict[str, list[MaskEvaluation]] = defaultdict(list)
    for item in evaluations:
        grouped[item.object_size].append(item)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    for item in evaluations:
        labels = sample_by_id[item.sample_id].meta["instance_labels"]
        label = labels.get(item.object_id, labels.get(str(item.object_id), "Unknown"))
        by_class[str(label)].append(item)
    return {
        "metric_implementation": METRIC_IMPLEMENTATION,
        "failure_policy_version": policy.version,
        "mask_count": len(evaluations),
        "miou": mean("iou"),
        "dice": mean("dice"),
        "pixel_precision": mean("pixel_precision"),
        "pixel_recall": mean("pixel_recall"),
        "boundary_iou": mean("boundary_iou"),
        "boundary_f_score": mean("boundary_f_score"),
        "mask_failure_rate_ratio": float(np.mean([item.failed for item in evaluations])),
        "mask_failure_rate_pct": float(np.mean([item.failed for item in evaluations]) * 100),
        "miou_by_size": {size: float(np.mean([item.iou for item in values])) for size, values in grouped.items()},
        "miou_by_class": {label: float(np.mean([item.iou for item in values])) for label, values in by_class.items()},
        "failures": [item.as_dict() for item in evaluations],
    }


class SegmentationEvaluator:
    """Person C evaluator plugin for D's protocol-locked benchmark runner."""

    task = "segmentation"
    metric_versions = {
        "miou": METRIC_IMPLEMENTATION,
        "boundary_iou": METRIC_IMPLEMENTATION,
        "mask_failure_rate": METRIC_IMPLEMENTATION,
    }

    def __init__(self, *, policy: MaskFailurePolicy = MaskFailurePolicy()) -> None:
        self.policy = policy

    def evaluate(
        self,
        predictions: Sequence[ModelPrediction],
        samples: Sequence[Sample],
        protocol: BenchmarkProtocol,
    ) -> EvaluationResult:
        segmentation = tuple(item for item in predictions if isinstance(item, SegmentationPrediction))
        if len(segmentation) != len(predictions):
            raise TypeError("SegmentationEvaluator accepts SegmentationPrediction values only")
        suite = segmentation_metric_suite(segmentation, samples, policy=self.policy)
        evaluations = [
            item
            for sample in samples
            for item in evaluate_prediction(
                sample,
                next(prediction for prediction in segmentation if prediction.sample_id == sample.sample_id),
                policy=self.policy,
            )
        ]
        headline = _metric("miou", suite["miou"], higher_is_better=True)
        supplemental = (
            _metric("boundary_iou", suite["boundary_iou"], higher_is_better=True),
            _metric("mask_failure_rate", suite["mask_failure_rate_ratio"], higher_is_better=False),
        )
        per_sample: dict[str, tuple[MetricEnvelope, ...]] = {}
        for sample in samples:
            own = [item for item in evaluations if item.sample_id == sample.sample_id]
            if own:
                per_sample[sample.sample_id] = (
                    _metric("miou", float(np.mean([item.iou for item in own])), higher_is_better=True),
                    _metric("boundary_iou", float(np.mean([item.boundary_iou for item in own])), higher_is_better=True),
                    _metric("mask_failure_rate", float(np.mean([item.failed for item in own])), higher_is_better=False),
                )
        failures = tuple(
            FailureCase(
                case_id=f"{protocol.protocol_id}:{item.sample_id}:{item.object_id}",
                sample_id=item.sample_id,
                model_id="sam2",
                protocol_id=protocol.protocol_id,
                clean_metrics=(),
                attacked_metrics=(
                    _metric("miou", item.iou, higher_is_better=True),
                    _metric("boundary_iou", item.boundary_iou, higher_is_better=True),
                ),
                reason=item.failure_reason,
                affected_object_id=item.object_id,
                metadata={"failure_policy_version": self.policy.version, "object_size": item.object_size},
            )
            for item in evaluations
            if item.failed
        )
        return EvaluationResult(
            task="segmentation",
            protocol_id=protocol.protocol_id,
            headline=headline,
            supplemental_metrics=supplemental,
            per_sample_metrics=per_sample,
            failures=failures,
        )


def _metric(name: str, value: float, *, higher_is_better: bool) -> MetricEnvelope:
    return MetricEnvelope(
        name=name,
        value=value,
        unit="ratio",
        percent_value=value * 100.0,
        version=METRIC_IMPLEMENTATION,
        higher_is_better=higher_is_better,
    )


def _failure_reason(
    predicted: np.ndarray,
    instance_mask: np.ndarray,
    target_id: int,
    iou: float,
    boundary: float,
    policy: MaskFailurePolicy,
) -> FailureReason:
    if not predicted.any():
        return "empty_prediction"
    overlaps = instance_mask[predicted]
    foreign = int(np.logical_and(predicted, np.logical_and(instance_mask > 0, instance_mask != target_id)).sum())
    if foreign > int(np.logical_and(predicted, instance_mask == target_id).sum()):
        return "wrong_object"
    components = _component_count(predicted)
    if components >= policy.severe_component_count:
        return "split_severe"
    merged_instances = {int(value) for value in overlaps if value > 0 and value != target_id}
    if len(merged_instances) >= 2:
        return "merge_severe"
    if iou < policy.iou_threshold:
        return "iou_below_threshold"
    if boundary < policy.boundary_iou_threshold:
        return "boundary_error"
    return "correct"


def _binary_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("segmentation masks must be same-shape two-dimensional arrays")
    return predicted, truth


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        interior &= padded[1 + dy : 1 + dy + mask.shape[0], 1 + dx : 1 + dx + mask.shape[1]]
    return mask & ~interior


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    output = mask.copy()
    for _ in range(max(0, radius)):
        padded = np.pad(output, 1, constant_values=False)
        output = np.zeros_like(output)
        for dy in range(3):
            for dx in range(3):
                output |= padded[dy : dy + output.shape[0], dx : dx + output.shape[1]]
    return output


def _component_count(mask: np.ndarray) -> int:
    pending = set(map(tuple, np.argwhere(mask)))
    components = 0
    while pending:
        components += 1
        stack = [pending.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in pending:
                    pending.remove(neighbor)
                    stack.append(neighbor)
    return components
