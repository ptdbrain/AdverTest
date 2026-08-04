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
SensorKind = Literal["image", "camera_rig", "lidar"]
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

# Category is attached to individual attacks where group membership is too
# coarse. Group A has both noise and weather-like corruptions.
ATTACK_CATEGORY: dict[str, str] = {
    "fog": "weather",
    "frost": "weather",
    "snow": "weather",
    "depth_fog": "weather",
    "depth_rain": "weather",
    "depth_snow": "weather",
    "lidar_fog": "weather",
    "lidar_snow": "weather",
    "brightness": "weather",
    "contrast": "weather",
    "saturate": "weather",
    "gaussian_noise": "noise",
    "shot_noise": "noise",
    "impulse_noise": "noise",
    "speckle_noise": "noise",
    "jpeg_compression": "noise",
    "pixelate": "noise",
    "gaussian_blur": "noise",
    "defocus_blur": "noise",
    "motion_blur": "noise",
    "zoom_blur": "noise",
    "glass_blur": "noise",
    "elastic_transform": "noise",
    "spatter": "noise",
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


@dataclass(frozen=True, slots=True)
class Box3D:
    """Minimal sensor-frame 3D cuboid used by multimodal attacks."""

    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    label: str
    score: float = 1.0
    vx: float = 0.0
    vy: float = 0.0
    native_label: str | None = None


@dataclass(frozen=True, slots=True)
class CameraView:
    """One calibrated camera view and an optional previous keyframe."""

    name: str
    image: np.ndarray
    depth: np.ndarray | None = None
    intrinsic: np.ndarray | None = None
    sensor_to_ego: np.ndarray | None = None
    previous_image: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class LidarFrame:
    """Point cloud with named columns; ``ring`` is required for beam attacks."""

    points: np.ndarray
    fields: tuple[str, ...] = ("x", "y", "z", "intensity", "ring")
    sensor_model: str = "unknown"

    def column(self, name: str) -> np.ndarray:
        try:
            index = self.fields.index(name)
        except ValueError as exc:
            raise ValueError(f"LiDAR frame has no {name!r} field") from exc
        return self.points[:, index]


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
    camera_views: tuple[CameraView, ...] = ()
    lidar_frame: LidarFrame | None = None
    boxes3d: tuple[Box3D, ...] = ()
    anonymized: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.image.shape  # type: ignore[return-value]

    def with_image(self, image: np.ndarray) -> Sample:
        """Return a copy carrying new pixels; ground truth is never modified."""
        return replace(self, image=image)

    def with_camera_views(self, views: tuple[CameraView, ...] | list[CameraView]) -> Sample:
        """Return a copy with views replaced and ``image`` synced to CAM_FRONT."""
        normalized = tuple(views)
        front = next((view.image for view in normalized if view.name == "CAM_FRONT"), self.image)
        return replace(self, image=front, camera_views=normalized)

    def with_lidar_frame(self, frame: LidarFrame | None) -> Sample:
        """Return a copy carrying an attacked multimodal point cloud."""
        return replace(self, lidar_frame=frame)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Model output for a single sample, already post-processed."""

    sample_id: str
    boxes: tuple[Box, ...] = ()
    boxes3d: tuple[Box3D, ...] = ()
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
    capabilities: frozenset[str] = frozenset()
    checkpoint_hash: str | None = None
    preprocessing_version: str = "default"
    runnable: bool = True


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
