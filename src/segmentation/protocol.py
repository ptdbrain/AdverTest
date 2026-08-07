"""Dataset and annotation rules owned by the SAM2 segmentation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from src.core.types import Sample

SEGMENTATION_CLASSES = (
    "Person",
    "Rider",
    "Car",
    "Truck",
    "Bus",
    "Bicycle",
    "Motorcycle",
)
CITYSCAPES_IN_DOMAIN = "cityscapes-reviewed-instance-v1"
BDD100K_EXTERNAL_ONLY = "bdd100k-segmentation-external-v1"
DATASET_ROLE = Literal["train", "validation", "locked_test", "external"]

# Maps native Cityscapes/BDD labels to the canonical protocol.  The broader
# display group is intentionally separate from the scientific class.
CLASS_MAPPING = {
    "person": "Person",
    "rider": "Rider",
    "car": "Car",
    "truck": "Truck",
    "bus": "Bus",
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
}
STORY_GROUP = {
    "Person": "Pedestrian",
    "Rider": "Cyclist-Rider",
    "Car": "Vehicle",
    "Truck": "Vehicle",
    "Bus": "Vehicle",
    "Bicycle": "Cyclist-Rider",
    "Motorcycle": "Cyclist-Rider",
}


@dataclass(frozen=True, slots=True)
class MaskValidationResult:
    instance_ids: tuple[int, ...]
    object_sizes: dict[int, str]
    reviewed: bool
    valid_region_fraction: float


@dataclass(frozen=True, slots=True)
class GroundTruthBoxPrompt:
    """Fixed GT-box prompt persisted in the SAM benchmark manifest."""

    prompt_id: str
    object_id: int
    coordinates: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.coordinates
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ground-truth box prompt must satisfy x1 < x2 and y1 < y2")


def fixed_box_prompts(sample: Sample) -> tuple[GroundTruthBoxPrompt, ...]:
    """Read immutable GT-box prompts supplied by ingestion/BenchmarkProtocol.

    The dataset platform persists these records under ``meta.sam_prompts``;
    deriving a new prompt from a detector prediction is intentionally forbidden.
    """
    raw = sample.meta.get("sam_prompts")
    if not isinstance(raw, (tuple, list)):
        raise ValueError("SAM sample is missing fixed ground-truth box prompts")
    prompts: list[GroundTruthBoxPrompt] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("sam_prompts entries must be dictionaries")
        coordinates = tuple(float(value) for value in item.get("coordinates", ()))
        if len(coordinates) != 4:
            raise ValueError("SAM box prompt requires four coordinates")
        prompts.append(
            GroundTruthBoxPrompt(
                prompt_id=str(item["prompt_id"]),
                object_id=int(item["object_id"]),
                coordinates=coordinates,  # type: ignore[arg-type]
            )
        )
    if not prompts:
        raise ValueError("SAM sample requires at least one fixed ground-truth box prompt")
    return tuple(prompts)


def object_size_bucket(area_pixels: int) -> str:
    """COCO-like size buckets, stable across metric and training reports."""
    if area_pixels < 32**2:
        return "small"
    if area_pixels < 96**2:
        return "medium"
    return "large"


def validate_segmentation_sample(
    sample: Sample,
    *,
    role: DATASET_ROLE,
    allow_unreviewed_training: bool = False,
) -> MaskValidationResult:
    """Validate reviewed integer instance masks before SAM benchmark/training.

    ``sample.mask`` uses 0 as background and positive integer instance IDs.
    ``meta.instance_labels`` must map each instance ID to a canonical class.
    Pseudo masks are allowed only as explicitly marked auxiliary training data.
    """
    mask = sample.mask
    if mask is None or mask.ndim != 2 or mask.shape != sample.image.shape[:2]:
        raise ValueError("segmentation sample needs a 2-D mask aligned with its image")
    if not np.isfinite(mask).all() or np.any(mask < 0):
        raise ValueError("instance mask must contain finite non-negative IDs")
    if not np.allclose(mask, np.round(mask)):
        raise ValueError("instance mask must contain integer instance IDs")
    ids = tuple(int(value) for value in np.unique(mask) if value > 0)
    if not ids:
        raise ValueError("instance mask has no foreground object")
    meta: dict[str, Any] = sample.meta
    reviewed = bool(meta.get("mask_reviewed", False))
    source = str(meta.get("mask_source", "unknown"))
    if role in {"validation", "locked_test", "external"} and not reviewed:
        raise ValueError(f"{role} masks must be human-reviewed")
    if role == "train" and not reviewed and not (allow_unreviewed_training and source == "pseudo"):
        raise ValueError("unreviewed masks are permitted only as explicitly enabled pseudo training data")
    labels = meta.get("instance_labels", {})
    normalized = {int(key): value for key, value in labels.items()}
    missing = [instance_id for instance_id in ids if normalized.get(instance_id) not in SEGMENTATION_CLASSES]
    if missing:
        raise ValueError(f"missing/invalid canonical labels for instance IDs: {missing}")
    valid_region = meta.get("valid_region")
    if valid_region is None:
        fraction = 1.0
    else:
        valid = np.asarray(valid_region, dtype=bool)
        if valid.shape != mask.shape:
            raise ValueError("valid_region must align with the instance mask")
        fraction = float(valid.mean())
        if fraction <= 0:
            raise ValueError("valid_region must retain at least one pixel")
    return MaskValidationResult(
        instance_ids=ids,
        object_sizes={instance_id: object_size_bucket(int((mask == instance_id).sum())) for instance_id in ids},
        reviewed=reviewed,
        valid_region_fraction=fraction,
    )
