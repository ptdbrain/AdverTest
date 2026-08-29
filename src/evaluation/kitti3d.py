"""KITTI 3D object detection evaluator.

Computes macro-averaged BEV AP50 over classes present in the ground truth,
distance-bucketed AP50 (near, medium, far), per-sample diagnostics, and failure case detection.
Implements the ``TaskEvaluator`` protocol consumed by the generic benchmark runner.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from collections.abc import Sequence

from src.core.types import Box3D, DetectionPrediction, ModelPrediction, Sample
from src.evaluation.base import EvaluationResult
from src.evaluation.contracts import FailureCase, MetricEnvelope
from src.evaluation.geometry3d import bev_iou, match_boxes3d
from src.pipeline.protocol import BenchmarkProtocol


def _box_distance(box: Box3D) -> float:
    """Compute horizontal BEV Euclidean distance to box center from sensor origin."""
    return float(math.sqrt(box.x**2 + box.y**2))


def _distance_bucket(dist: float) -> str:
    """Classify horizontal distance into near (<20m), medium (20-40m), or far (>=40m)."""
    if dist < 20.0:
        return "near"
    if dist < 40.0:
        return "medium"
    return "far"


class Kitti3DEvaluator:
    """BEV-based 3D object detection evaluator for KITTI-style datasets."""

    task = "detection3d"
    metric_versions = {"kitti_3d_ap": "advertest-bev-v1", "bev_iou": "1.0.0"}

    def __init__(self, *, iou_threshold: float = 0.5) -> None:
        self._iou_threshold = iou_threshold

    def evaluate(
        self,
        predictions: Sequence[ModelPrediction],
        samples: Sequence[Sample],
        protocol: BenchmarkProtocol,
    ) -> EvaluationResult:
        """Evaluate 3D detection predictions against ground truth.

        Args:
            predictions: One DetectionPrediction per sample.
            samples: Ground-truth samples with ``boxes3d`` in LiDAR frame.
            protocol: Locked benchmark protocol for identity tracking.

        Returns:
            EvaluationResult with headline AP, distance-bucketed AP,
            supplemental metrics, per-sample diagnostics, and failure cases.

        Raises:
            ValueError: If predictions/samples lengths or IDs mismatch,
                        or if predictions are not DetectionPrediction.
        """
        if len(predictions) != len(samples):
            raise ValueError(
                f"prediction count ({len(predictions)}) != sample count ({len(samples)})"
            )

        per_sample_metrics: dict[str, tuple[MetricEnvelope, ...]] = {}
        failures: list[FailureCase] = []

        # Per-class accumulators for global AP calculation
        class_tp_scores: dict[str, list[float]] = defaultdict(list)
        class_fp_scores: dict[str, list[float]] = defaultdict(list)
        class_gt_counts: dict[str, int] = defaultdict(int)

        # Distance-bucketed accumulators
        bucket_tp_scores: dict[str, dict[str, list[float]]] = {
            "near": defaultdict(list),
            "medium": defaultdict(list),
            "far": defaultdict(list),
        }
        bucket_fp_scores: dict[str, dict[str, list[float]]] = {
            "near": defaultdict(list),
            "medium": defaultdict(list),
            "far": defaultdict(list),
        }
        bucket_gt_counts: dict[str, dict[str, int]] = {
            "near": defaultdict(int),
            "medium": defaultdict(int),
            "far": defaultdict(int),
        }

        total_iou_sum = 0.0
        total_matched_count = 0
        total_tp_count = 0
        total_fp_count = 0
        total_gt_count = 0

        for prediction, sample in zip(predictions, samples, strict=False):
            if prediction.sample_id != sample.sample_id:
                raise ValueError(
                    f"sample ID mismatch: prediction={prediction.sample_id!r}, "
                    f"sample={sample.sample_id!r}"
                )
            if not isinstance(prediction, DetectionPrediction):
                raise ValueError(
                    f"expected DetectionPrediction, got {type(prediction).__name__}"
                )

            ground_truths = sample.boxes3d or ()
            predicted_boxes = prediction.boxes3d or ()

            # Accumulate per-class and distance-bucketed GT counts
            for gt_box in ground_truths:
                class_gt_counts[gt_box.label] += 1
                gt_bucket = _distance_bucket(_box_distance(gt_box))
                bucket_gt_counts[gt_bucket][gt_box.label] += 1

            # Greedy BEV matching
            matches = match_boxes3d(
                ground_truths, predicted_boxes, iou_threshold=self._iou_threshold
            )
            matched_gt_indices = {m.gt_index for m in matches}
            matched_pred_indices = {m.prediction_index for m in matches}

            # Per-match scoring
            sample_iou_sum = 0.0
            for match in matches:
                matched_pred = predicted_boxes[match.prediction_index]
                pred_dist = _box_distance(matched_pred)
                pred_bucket = _distance_bucket(pred_dist)

                class_tp_scores[matched_pred.label].append(matched_pred.score)
                bucket_tp_scores[pred_bucket][matched_pred.label].append(matched_pred.score)
                sample_iou_sum += match.iou
                total_iou_sum += match.iou
                total_matched_count += 1

            # Unmatched predictions -> false positives
            for pred_idx, pred_box in enumerate(predicted_boxes):
                if pred_idx in matched_pred_indices:
                    continue
                pred_dist = _box_distance(pred_box)
                pred_bucket = _distance_bucket(pred_dist)
                class_fp_scores[pred_box.label].append(pred_box.score)
                bucket_fp_scores[pred_bucket][pred_box.label].append(pred_box.score)

            mean_sample_iou = (
                sample_iou_sum / len(matches) if matches else 0.0
            )

            # --- Failure detection ---
            _detect_failures(
                sample=sample,
                ground_truths=ground_truths,
                predicted_boxes=predicted_boxes,
                matched_gt_indices=matched_gt_indices,
                matched_pred_indices=matched_pred_indices,
                iou_threshold=self._iou_threshold,
                protocol_id=protocol.protocol_id,
                task=self.task,
                failures=failures,
            )

            # --- Per-sample metrics ---
            per_sample_metrics[sample.sample_id] = _build_per_sample_metrics(
                gt_count=len(ground_truths),
                pred_count=len(predicted_boxes),
                matched_count=len(matches),
                mean_bev_iou=mean_sample_iou,
                bev_iou_version=self.metric_versions["bev_iou"],
            )

        # --- Global AP computation ---
        total_gt_count = sum(class_gt_counts.values())
        total_tp_count = sum(len(v) for v in class_tp_scores.values())
        total_fp_count = sum(len(v) for v in class_fp_scores.values())

        kitti_3d_ap = _macro_average_precision(
            class_tp_scores, class_fp_scores, class_gt_counts
        )
        ap_near = _macro_average_precision(
            bucket_tp_scores["near"], bucket_fp_scores["near"], bucket_gt_counts["near"]
        )
        ap_medium = _macro_average_precision(
            bucket_tp_scores["medium"], bucket_fp_scores["medium"], bucket_gt_counts["medium"]
        )
        ap_far = _macro_average_precision(
            bucket_tp_scores["far"], bucket_fp_scores["far"], bucket_gt_counts["far"]
        )

        global_mean_bev_iou = (
            total_iou_sum / total_matched_count if total_matched_count > 0 else 0.0
        )
        global_precision = (
            total_tp_count / (total_tp_count + total_fp_count)
            if (total_tp_count + total_fp_count) > 0
            else 0.0
        )
        global_recall = (
            total_tp_count / total_gt_count if total_gt_count > 0 else 0.0
        )

        headline = MetricEnvelope(
            name="kitti_3d_ap",
            value=kitti_3d_ap,
            unit="ratio",
            percent_value=round(kitti_3d_ap * 100.0, 10),
            version=self.metric_versions["kitti_3d_ap"],
            higher_is_better=True,
        )

        supplemental = (
            MetricEnvelope(
                name="kitti_3d_ap_near",
                value=ap_near,
                unit="ratio",
                percent_value=round(ap_near * 100.0, 10),
                version=self.metric_versions["kitti_3d_ap"],
                higher_is_better=True,
            ),
            MetricEnvelope(
                name="kitti_3d_ap_medium",
                value=ap_medium,
                unit="ratio",
                percent_value=round(ap_medium * 100.0, 10),
                version=self.metric_versions["kitti_3d_ap"],
                higher_is_better=True,
            ),
            MetricEnvelope(
                name="kitti_3d_ap_far",
                value=ap_far,
                unit="ratio",
                percent_value=round(ap_far * 100.0, 10),
                version=self.metric_versions["kitti_3d_ap"],
                higher_is_better=True,
            ),
            MetricEnvelope(
                name="mean_bev_iou",
                value=global_mean_bev_iou,
                unit="ratio",
                percent_value=round(global_mean_bev_iou * 100.0, 10),
                version=self.metric_versions["bev_iou"],
                higher_is_better=True,
            ),
            MetricEnvelope(
                name="detection3d_precision",
                value=global_precision,
                unit="ratio",
                percent_value=round(global_precision * 100.0, 10),
                version="1.0.0",
                higher_is_better=True,
            ),
            MetricEnvelope(
                name="detection3d_recall",
                value=global_recall,
                unit="ratio",
                percent_value=round(global_recall * 100.0, 10),
                version="1.0.0",
                higher_is_better=True,
            ),
        )

        return EvaluationResult(
            task=self.task,
            protocol_id=protocol.protocol_id,
            headline=headline,
            supplemental_metrics=supplemental,
            per_sample_metrics=per_sample_metrics,
            failures=tuple(failures),
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_per_sample_metrics(
    *,
    gt_count: int,
    pred_count: int,
    matched_count: int,
    mean_bev_iou: float,
    bev_iou_version: str,
) -> tuple[MetricEnvelope, ...]:
    """Build the standard per-sample metric tuple."""
    return (
        MetricEnvelope(
            name="gt_count",
            value=float(gt_count),
            unit="count",
            percent_value=None,
            version="1.0.0",
            higher_is_better=False,
        ),
        MetricEnvelope(
            name="prediction_count",
            value=float(pred_count),
            unit="count",
            percent_value=None,
            version="1.0.0",
            higher_is_better=False,
        ),
        MetricEnvelope(
            name="matched_count",
            value=float(matched_count),
            unit="count",
            percent_value=None,
            version="1.0.0",
            higher_is_better=True,
        ),
        MetricEnvelope(
            name="missed_count",
            value=float(gt_count - matched_count),
            unit="count",
            percent_value=None,
            version="1.0.0",
            higher_is_better=False,
        ),
        MetricEnvelope(
            name="false_positive_count",
            value=float(pred_count - matched_count),
            unit="count",
            percent_value=None,
            version="1.0.0",
            higher_is_better=False,
        ),
        MetricEnvelope(
            name="mean_bev_iou",
            value=mean_bev_iou,
            unit="ratio",
            percent_value=round(mean_bev_iou * 100.0, 10),
            version=bev_iou_version,
            higher_is_better=True,
        ),
    )


def _detect_failures(
    *,
    sample: Sample,
    ground_truths: tuple[Box3D, ...],
    predicted_boxes: tuple[Box3D, ...],
    matched_gt_indices: set[int],
    matched_pred_indices: set[int],
    iou_threshold: float,
    protocol_id: str,
    task: str,
    failures: list[FailureCase],
) -> None:
    """Detect and append failure cases for one sample with distance metadata."""
    # Missed ground truths
    for gt_idx, gt_box in enumerate(ground_truths):
        if gt_idx in matched_gt_indices:
            continue
        dist = _box_distance(gt_box)
        failures.append(
            FailureCase(
                case_id=f"fail-{uuid.uuid4().hex[:16]}",
                sample_id=sample.sample_id,
                model_id="unknown",
                protocol_id=protocol_id,
                clean_metrics=(),
                attacked_metrics=(),
                reason="MISSED_GT",
                metadata={
                    "task": task,
                    "failure_type": "MISSED_GT",
                    "label": gt_box.label,
                    "distance": round(dist, 2),
                    "distance_bucket": _distance_bucket(dist),
                },
            )
        )

    # Unmatched predictions
    for pred_idx, pred_box in enumerate(predicted_boxes):
        if pred_idx in matched_pred_indices:
            continue

        best_iou_value = 0.0
        best_gt_label: str | None = None
        for gt_box in ground_truths:
            overlap = bev_iou(gt_box, pred_box)
            if overlap > best_iou_value:
                best_iou_value = overlap
                best_gt_label = gt_box.label

        if best_iou_value > 0 and best_gt_label and best_gt_label != pred_box.label:
            reason = "CLASS_MISMATCH"
        elif 0 < best_iou_value < iou_threshold:
            reason = "LOCALIZATION_FAILURE"
        else:
            reason = "FALSE_POSITIVE"

        dist = _box_distance(pred_box)
        failures.append(
            FailureCase(
                case_id=f"fail-{uuid.uuid4().hex[:16]}",
                sample_id=sample.sample_id,
                model_id="unknown",
                protocol_id=protocol_id,
                clean_metrics=(),
                attacked_metrics=(),
                reason=reason,
                metadata={
                    "task": task,
                    "failure_type": reason,
                    "label": pred_box.label,
                    "distance": round(dist, 2),
                    "distance_bucket": _distance_bucket(dist),
                },
            )
        )


def _macro_average_precision(
    class_tp_scores: dict[str, list[float]],
    class_fp_scores: dict[str, list[float]],
    class_gt_counts: dict[str, int],
) -> float:
    """Compute 11-point interpolated AP, macro-averaged over classes in GT.

    Uses the standard KITTI-style 11-point interpolation (R=0.0, 0.1, ..., 1.0).
    """
    ap_accumulator = 0.0
    evaluated_class_count = 0

    for label, gt_count in class_gt_counts.items():
        if gt_count == 0:
            continue

        scored_hits = [(score, True) for score in class_tp_scores.get(label, [])]
        scored_hits += [(score, False) for score in class_fp_scores.get(label, [])]
        scored_hits.sort(key=lambda item: item[0], reverse=True)

        if not scored_hits:
            evaluated_class_count += 1
            continue

        tp_cumulative = 0
        fp_cumulative = 0
        recall_values: list[float] = []
        precision_values: list[float] = []

        for _score, is_tp in scored_hits:
            if is_tp:
                tp_cumulative += 1
            else:
                fp_cumulative += 1
            recall_values.append(tp_cumulative / gt_count)
            precision_values.append(tp_cumulative / (tp_cumulative + fp_cumulative))

        # 11-point interpolation
        class_ap = 0.0
        for recall_threshold_idx in range(11):
            recall_threshold = recall_threshold_idx / 10.0
            interpolated_precision = 0.0
            for recall_val, precision_val in zip(recall_values, precision_values, strict=False):
                if recall_val >= recall_threshold:
                    interpolated_precision = max(interpolated_precision, precision_val)
            class_ap += interpolated_precision / 11.0

        ap_accumulator += class_ap
        evaluated_class_count += 1

    return ap_accumulator / evaluated_class_count if evaluated_class_count > 0 else 0.0
