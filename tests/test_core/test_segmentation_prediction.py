from __future__ import annotations

import numpy as np
import pytest

from src.core.types import MaskPrediction, SegmentationPrediction


def test_segmentation_prediction_carries_fixed_prompt_id_and_instances() -> None:
    prediction = SegmentationPrediction(
        sample_id="sample",
        prompt_id="gt-box:sample:1",
        instances=(MaskPrediction(instance_id="1", mask=np.ones((4, 4), dtype=np.bool_), score=0.9),),
    )
    assert prediction.prompt_id == "gt-box:sample:1"


def test_mask_prediction_rejects_non_boolean_masks() -> None:
    with pytest.raises(ValueError, match="bool dtype"):
        MaskPrediction(instance_id="1", mask=np.ones((4, 4), dtype=np.uint8))
