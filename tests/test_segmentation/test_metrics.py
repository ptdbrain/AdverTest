from __future__ import annotations

import numpy as np
import pytest

from src.core.types import MaskPrediction, Sample, SegmentationPrediction
from src.evaluation.segmentation_metrics import binary_iou, evaluate_prediction, segmentation_metric_suite


def _sample() -> Sample:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 2:7] = 1
    return Sample(
        "s1", np.zeros((10, 10, 3), dtype=np.float32), mask=mask,
        meta={"mask_reviewed": True, "mask_source": "human", "instance_labels": {1: "Car"}},
    )


def _prediction(mask: np.ndarray) -> SegmentationPrediction:
    return SegmentationPrediction(
        sample_id="s1", prompt_id="gt-box:s1:1",
        instances=(MaskPrediction(instance_id="1", mask=mask.astype(np.bool_), score=0.9),),
    )


def test_perfect_mask_metrics_are_one() -> None:
    sample = _sample()
    predicted = sample.mask == 1
    result = evaluate_prediction(sample, _prediction(predicted))[0]
    assert result.iou == pytest.approx(1.0)
    assert result.boundary_iou == pytest.approx(1.0)
    assert result.failure_reason == "correct"


def test_empty_prediction_is_a_versioned_failure() -> None:
    result = evaluate_prediction(_sample(), _prediction(np.zeros((10, 10), dtype=np.uint8)))[0]
    assert result.failure_reason == "empty_prediction"


def test_metric_suite_exposes_ratio_and_percent_units() -> None:
    sample = _sample()
    suite = segmentation_metric_suite([_prediction(sample.mask == 1)], [sample])
    assert suite["miou"] == pytest.approx(1.0)
    assert suite["mask_failure_rate_ratio"] == 0.0
    assert suite["mask_failure_rate_pct"] == 0.0


def test_binary_iou_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        binary_iou(np.zeros((2, 2)), np.zeros((3, 3)))
