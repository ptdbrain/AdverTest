"""SAM2-specific protocol and reviewed-mask validation utilities."""

from src.segmentation.protocol import (
    BDD100K_EXTERNAL_ONLY,
    CITYSCAPES_IN_DOMAIN,
    GroundTruthBoxPrompt,
    SEGMENTATION_CLASSES,
    fixed_box_prompts,
    validate_segmentation_sample,
)

__all__ = [
    "BDD100K_EXTERNAL_ONLY",
    "CITYSCAPES_IN_DOMAIN",
    "GroundTruthBoxPrompt",
    "SEGMENTATION_CLASSES",
    "fixed_box_prompts",
    "validate_segmentation_sample",
]
