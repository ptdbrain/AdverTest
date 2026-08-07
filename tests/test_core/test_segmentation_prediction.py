from __future__ import annotations

import numpy as np
import pytest

from src.core.types import SegmentationPrediction, SegmentationPrompt


def test_segmentation_prediction_requires_prompt_aligned_outputs() -> None:
    prompt = SegmentationPrompt("box", (1, 1, 4, 4), object_id=1)
    prediction = SegmentationPrediction("sample", "sam-b0", (np.ones((4, 4)),), (0.9,), (prompt,))
    assert prediction.prompts[0].object_id == 1


def test_segmentation_prediction_rejects_misaligned_outputs() -> None:
    with pytest.raises(ValueError, match="identical lengths"):
        SegmentationPrediction("sample", "sam-b0", (), (0.9,), ())


def test_box_prompt_is_validated() -> None:
    with pytest.raises(ValueError, match="x1 < x2"):
        SegmentationPrompt("box", (4, 1, 1, 4), object_id=1)
