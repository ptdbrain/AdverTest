"""The reference adapter must be a usable baseline, not a stub that always agrees."""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.base import GradientsNotSupportedError, ModelAdapter
from src.core.types import CLASSES, Sample
from src.evaluation.detection_metrics import average_precision


def test_metadata_declares_gradient_support(adapter: ModelAdapter) -> None:
    info = adapter.metadata()
    assert info.supports_gradients is True
    assert info.task == "detection2d"
    assert "thr" in info.version, "version must change with the threshold, for cache correctness"


def test_clean_detection_is_accurate(adapter: ModelAdapter, samples: list[Sample]) -> None:
    predictions = adapter.predict(samples)
    assert average_precision(predictions, samples) >= 0.8


def test_predictions_use_the_normalised_label_space(adapter: ModelAdapter, samples: list[Sample]) -> None:
    for prediction in adapter.predict(samples):
        for box in prediction.boxes:
            assert box.label in CLASSES
            assert 0.0 <= box.score <= 1.0


def test_gradient_shape_and_direction(adapter: ModelAdapter, sample: Sample) -> None:
    gradient = adapter.input_gradient(sample)
    assert gradient.shape == sample.image.shape
    assert gradient.dtype == np.float32
    assert np.any(gradient != 0.0), "gradient must be non-zero where objects are"
    # Ascending the loss must make the objects harder to see.
    before = adapter.loss_for_attack(sample)
    after = adapter.loss_for_attack(sample.with_image(np.clip(sample.image + 0.02 * np.sign(gradient), 0, 1)))
    assert after > before


def test_black_box_adapter_reports_missing_gradients() -> None:
    class BlackBox(ModelAdapter):
        name = "test_black_box"

        def predict(self, samples):
            return []

        def metadata(self):
            raise NotImplementedError

    with pytest.raises(GradientsNotSupportedError):
        BlackBox().input_gradient(Sample("s", np.zeros((4, 4, 3), dtype=np.float32)))
