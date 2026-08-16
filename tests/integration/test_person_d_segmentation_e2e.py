import numpy as np

from src.core.contracts import MaskWireV1
from src.core.types import MaskPrediction, SegmentationPrediction


def test_segmentation_mask_prediction_and_wire_round_trip() -> None:
    mask = np.array([[True, False], [True, True]], dtype=np.bool_)
    prediction = SegmentationPrediction(
        sample_id="seg-1",
        prompt_id="fixed-box-v1",
        instances=(MaskPrediction(instance_id="m1", mask=mask),),
    )
    restored = MaskWireV1.from_array(prediction.instances[0].mask).to_array()
    assert np.array_equal(restored, mask)
    assert prediction.prompt_id == "fixed-box-v1"
