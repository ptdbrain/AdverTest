"""Optional RT-DETR adapter for black-box and transfer benchmarking.

Ultralytics exposes RT-DETR checkpoints through the same prediction API as
YOLO. This adapter deliberately exposes no gradients: it is a target model for
transfer evaluation, not a surrogate for Group D white-box generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from src.adapters import MODELS
from src.adapters.yolo11 import Yolo11Adapter
from src.core.types import ModelInfo


@MODELS.register
class RtdetrAdapter(Yolo11Adapter):
    """Ultralytics RT-DETR detector used as a black-box 2D target model."""

    name: ClassVar[str] = "rtdetr"
    version: ClassVar[str] = "ultralytics-rtdetr-transfer-v1"
    supports_gradients: ClassVar[bool] = False
    capabilities: ClassVar[frozenset[str]] = frozenset()
    owner: ClassVar[str] = "group-f"

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            task="detection2d",
            version=(
                f"{self.version}:{Path(self.weights).stem}:"
                f"imgsz{self.image_size}:conf{self.score_threshold:.3f}"
            ),
            supports_gradients=False,
        )
