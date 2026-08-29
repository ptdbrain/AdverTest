from __future__ import annotations

import numpy as np
import pytest

from src.core.types import Sample
from src.segmentation.protocol import validate_segmentation_sample


def _sample(*, reviewed: bool = True) -> Sample:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    return Sample(
        "s1",
        np.zeros((8, 8, 3), dtype=np.float32),
        mask=mask,
        meta={"mask_reviewed": reviewed, "mask_source": "human", "instance_labels": {1: "Car"}},
    )


def test_reviewed_mask_is_valid_for_locked_test() -> None:
    result = validate_segmentation_sample(_sample(), role="locked_test")
    assert result.instance_ids == (1,)
    assert result.object_sizes == {1: "small"}


def test_unreviewed_mask_is_rejected_for_locked_test() -> None:
    with pytest.raises(ValueError, match="human-reviewed"):
        validate_segmentation_sample(_sample(reviewed=False), role="locked_test")
