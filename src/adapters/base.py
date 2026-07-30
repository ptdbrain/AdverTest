"""Model adapter contract (plan §1.2).

Adding a model to AdverTest means writing one adapter — never touching the
core. The four functions from the plan map to:

===========================  ===========================================
Plan                         Method
===========================  ===========================================
``predict(batch)``           :meth:`ModelAdapter.predict`
``loss_for_attack(...)``     :meth:`ModelAdapter.loss_for_attack`
``postprocess()``            :meth:`ModelAdapter.postprocess`
``metadata()``               :meth:`ModelAdapter.metadata`
===========================  ===========================================

:meth:`ModelAdapter.input_gradient` is the extra bridge that keeps white-box
attacks framework-agnostic: the adapter owns the framework (torch, ONNX, …) and
hands back a plain numpy gradient, so an attack in ``src/attacks/adversarial/``
never imports torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from src.core.types import Box, Modality, ModelInfo, Prediction, Sample, Task


class GradientsNotSupportedError(NotImplementedError):
    """Raised when a white-box attack is pointed at a black-box adapter."""


class ModelAdapter(ABC):
    """Base class for every model under test."""

    name: ClassVar[str]
    task: ClassVar[Task] = "detection2d"
    version: ClassVar[str] = "0.1.0"
    modality: ClassVar[Modality] = "image"
    supports_gradients: ClassVar[bool] = False
    #: Team member responsible for this adapter (shown in the catalog).
    owner: ClassVar[str] = "unassigned"

    def __init__(self, *, score_threshold: float = 0.25, max_detections: int = 100) -> None:
        self.score_threshold = score_threshold
        self.max_detections = max_detections

    @abstractmethod
    def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
        """Run inference on a batch. Must return one prediction per sample."""

    @abstractmethod
    def metadata(self) -> ModelInfo:
        """Describe the model; ``version`` must change whenever weights change."""

    def postprocess(self, boxes: Sequence[Box]) -> tuple[Box, ...]:
        """Default post-processing: confidence filter + top-k cap."""
        kept = [box for box in boxes if box.score >= self.score_threshold and box.area > 0]
        kept.sort(key=lambda box: box.score, reverse=True)
        return tuple(kept[: self.max_detections])

    def loss_for_attack(self, sample: Sample, target: Any | None = None) -> float:
        """Scalar an untargeted attack *maximises* (higher = worse detection)."""
        raise GradientsNotSupportedError(f"{self.name} does not expose an attack loss")

    def input_gradient(self, sample: Sample, target: Any | None = None) -> np.ndarray:
        """``d loss_for_attack / d image`` with the same shape as the image."""
        raise GradientsNotSupportedError(
            f"{self.name} does not expose input gradients; use a black-box attack (group F)"
        )

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Catalog entry for the API / CLI."""
        return {
            "name": cls.name,
            "task": cls.task,
            "version": cls.version,
            "modality": cls.modality,
            "supports_gradients": cls.supports_gradients,
            "owner": cls.owner,
            "docstring": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
        }
