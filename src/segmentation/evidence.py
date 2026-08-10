"""SAM mask/boundary evidence serializer for the shared evidence boundary."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.types import Sample, SegmentationPrediction
from src.pipeline.evidence import prediction_payload


class SamSegmentationEvidenceSerializer:
    task = "segmentation"

    def write(self, *, root: Path, clean: Sample, attacked: Sample, clean_prediction: SegmentationPrediction, attacked_prediction: SegmentationPrediction) -> dict[str, str]:
        if clean.mask is None:
            raise ValueError("SAM evidence requires reviewed ground-truth instance masks")
        payload = {"ground_truth": _save(root / "ground_truth.png", clean.mask > 0), "clean_prediction": _save(root / "clean_prediction.png", _union(clean_prediction, clean.mask.shape)), "attacked_prediction": _save(root / "attacked_prediction.png", _union(attacked_prediction, clean.mask.shape))}
        (root / "segmentation_predictions.json").write_text(json.dumps({"clean": prediction_payload(clean_prediction), "attacked": prediction_payload(attacked_prediction)}, indent=2), encoding="utf-8")
        return payload


def _union(prediction: SegmentationPrediction, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for instance in prediction.instances:
        result |= instance.mask
    return result


def _save(path: Path, mask: np.ndarray) -> str:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return str(path)
