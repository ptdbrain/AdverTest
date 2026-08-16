"""Framework-agnostic building blocks and versioned handoff contracts."""

from src.core.contracts import (
    DetectionPredictionWire,
    MaskPredictionWire,
    MaskWireV1,
    ModelVersionMetadata,
    SegmentationPredictionWire,
)
from src.core.events import JobRequest, ProgressEvent

__all__ = [
    "DetectionPredictionWire",
    "JobRequest",
    "MaskPredictionWire",
    "MaskWireV1",
    "ModelVersionMetadata",
    "ProgressEvent",
    "SegmentationPredictionWire",
]
