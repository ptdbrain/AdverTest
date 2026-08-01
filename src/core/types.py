"""Domain types shared by every layer of AdverTest.

The whole platform speaks in terms of a small number of frozen dataclasses so
that attack plugins, model adapters, and the evaluator never depend on each
other's internals. Image contract (enforced by ``BaseAttack.run``):

    ``np.ndarray``, dtype ``float32``, shape ``(H, W, 3)``, values in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

Modality = Literal["image", "lidar", "multi"]
CostClass = Literal["cheap", "medium", "expensive"]
AttackGroup = Literal["A", "B", "C", "D", "E", "F"]
Task = Literal["detection2d", "segmentation", "detection3d"]

#: Label space normalised across KITTI / nuScenes (plan §1.1).
CLASSES: tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")

#: Severity 0 always means "no-op" (sanity check #1), 1..5 are the real levels.
MAX_SEVERITY = 5

GROUP_TITLES: dict[AttackGroup, str] = {
    "A": "Common corruptions",
    "B": "Physical weather (depth-aware)",
    "C": "Occlusion & sensor faults",
    "D": "Adversarial digital (white-box)",
    "E": "Adversarial patch (physical-plausible)",
    "F": "Black-box & transfer",
}

#: RobustScore aggregates groups into the four categories of plan §3 metric 13.
GROUP_CATEGORY: dict[AttackGroup, str] = {
    "A": "noise",
    "B": "weather",
    "C": "occlusion",
    "D": "adversarial",
    "E": "adversarial",
    "F": "adversarial",
}

#: Relative cost used by the pre-run GPU estimate (plan §5).
COST_WEIGHT: dict[CostClass, float] = {"cheap": 1.0, "medium": 4.0, "expensive": 20.0}


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned box in pixel coordinates, ``x1 < x2`` and ``y1 < y2``."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float = 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True, eq=False)
class Sample:
    """One evaluation unit: pixels, optional depth/LiDAR, and ground truth.

    ``eq=False`` because the payload holds numpy arrays, which have no
    unambiguous ``__eq__``. Use :func:`src.core.hashing.array_digest` to compare.
    """

    sample_id: str
    image: np.ndarray
    boxes: tuple[Box, ...] = ()
    mask: np.ndarray | None = None
    depth: np.ndarray | None = None
    lidar: np.ndarray | None = None
    anonymized: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape  # type: ignore[return-value]

    def with_image(self, image: np.ndarray) -> Sample:
        """Return a copy carrying new pixels; ground truth is never modified."""
        return replace(self, image=image)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Model output for a single sample, already post-processed."""

    sample_id: str
    boxes: tuple[Box, ...] = ()
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata every adapter must expose (plan §1.2, ``metadata()``)."""

    name: str
    task: Task
    version: str
    modality: Modality = "image"
    supports_gradients: bool = False
    classes: tuple[str, ...] = CLASSES


def validate_image(image: np.ndarray, *, like: np.ndarray | None = None) -> None:
    """Raise ``TypeError``/``ValueError`` when the image contract is broken."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}")
    if image.dtype != np.float32:
        raise ValueError(f"image must be float32, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape (H, W, 3), got {image.shape}")
    if like is not None and image.shape != like.shape:
        raise ValueError(f"attack changed image shape: {like.shape} -> {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError("image contains NaN or inf")
    low, high = float(image.min()), float(image.max())
    if low < -1e-6 or high > 1.0 + 1e-6:
        raise ValueError(f"image values must stay in [0, 1], got [{low:.4f}, {high:.4f}]")
