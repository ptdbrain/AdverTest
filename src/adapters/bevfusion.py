"""Optional MMDetection3D BEVFusion adapter (pinned 1.4.0 runtime)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.types import ModelInfo, Prediction, Sample


@MODELS.register
class BEVFusionAdapter(ModelAdapter):
    name = "bevfusion"
    task = "detection3d"
    modality = "multi"
    version = "mmdet3d-1.4.0"

    def __init__(self, *, config: str | None = None, weights: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.config, self.weights = config, weights
        try:
            import mmdet3d  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("BEVFusion requires MMDetection3D 1.4.0; install the 3D extra") from exc

    def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
        raise NotImplementedError("configure an MMDetection3D checkpoint/config before inference")

    def metadata(self) -> ModelInfo:
        return ModelInfo(self.name, self.task, self.version, self.modality, classes=("Car", "Pedestrian", "Cyclist"))
