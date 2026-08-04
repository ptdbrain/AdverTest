"""Metric maths on hand-checkable inputs."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.types import Box, Prediction, Sample
from src.evaluation.detection_metrics import (
    average_precision,
    detection_attack_success_rate,
    detection_summary,
    iou,
    match_boxes,
)
from src.evaluation.report import CellResult, RunReport


def _sample(boxes: tuple[Box, ...]) -> Sample:
    return Sample("s0", np.zeros((8, 8, 3), dtype=np.float32), boxes)


CAR = Box(0, 0, 10, 10, "Car")


def test_iou_identical_box_is_one() -> None:
    assert iou(CAR, CAR) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    assert iou(CAR, Box(20, 20, 30, 30, "Car")) == 0.0


def test_iou_half_overlap() -> None:
    # Intersection 5x10 = 50, union = 100 + 100 - 50 = 150.
    assert iou(CAR, Box(5, 0, 15, 10, "Car")) == pytest.approx(50 / 150)


def test_matching_ignores_class_mismatch() -> None:
    assert match_boxes([Box(0, 0, 10, 10, "Pedestrian", 0.9)], [CAR]) == {}


def test_perfect_prediction_scores_one() -> None:
    sample = _sample((CAR,))
    prediction = Prediction("s0", (Box(0, 0, 10, 10, "Car", 0.9),))
    assert average_precision([prediction], [sample]) == pytest.approx(1.0)


def test_one_of_two_objects_found_halves_ap() -> None:
    sample = _sample((CAR, Box(20, 20, 30, 30, "Car")))
    prediction = Prediction("s0", (Box(0, 0, 10, 10, "Car", 0.9),))
    assert average_precision([prediction], [sample]) == pytest.approx(0.5)


def test_low_scored_false_positive_does_not_erase_ap() -> None:
    """A confident hit plus a weak false alarm still yields AP 1.0."""
    sample = _sample((CAR,))
    prediction = Prediction("s0", (Box(0, 0, 10, 10, "Car", 0.9), Box(40, 40, 50, 50, "Car", 0.3)))
    assert average_precision([prediction], [sample]) == pytest.approx(1.0)


def test_no_predictions_scores_zero() -> None:
    assert average_precision([Prediction("s0")], [_sample((CAR,))]) == 0.0


def test_detection_summary_counts_tp_fp_and_fn() -> None:
    sample = _sample((CAR, Box(20, 20, 30, 30, "Car")))
    prediction = Prediction(
        "s0",
        (
            Box(0, 0, 10, 10, "Car", 0.9),
            Box(40, 40, 50, 50, "Car", 0.8),
        ),
    )
    summary = detection_summary([prediction], [sample])

    assert summary.true_positives == 1
    assert summary.false_positives == 1
    assert summary.false_negatives == 1
    assert summary.precision == pytest.approx(0.5)
    assert summary.recall == pytest.approx(0.5)


def test_attack_success_counts_clean_detections_lost_after_attack() -> None:
    sample = _sample((CAR, Box(20, 20, 30, 30, "Car")))
    clean = Prediction(
        "s0",
        (
            Box(0, 0, 10, 10, "Car", 0.9),
            Box(20, 20, 30, 30, "Car", 0.8),
        ),
    )
    attacked = Prediction("s0", (Box(0, 0, 10, 10, "Car", 0.7),))
    summary = detection_attack_success_rate([clean], [attacked], [sample])

    assert summary.clean_detected_truths == 2
    assert summary.lost_truths == 1
    assert summary.rate == pytest.approx(0.5)


def test_degradation_is_relative_to_clean_ap() -> None:
    report = RunReport("r0", "m", "m:1", "d", 4, ap_clean=0.8)
    cell = CellResult("gaussian_noise", "A", 3, ap=0.4, n_samples=4)
    assert report.degradation(cell) == pytest.approx(0.5)


def test_degradation_is_zero_when_baseline_is_zero() -> None:
    report = RunReport("r0", "m", "m:1", "d", 4, ap_clean=0.0)
    cell = CellResult("gaussian_noise", "A", 3, ap=0.0, n_samples=4)
    assert report.degradation(cell) == 0.0
