"""Task and model-family contracts used to construct adapters safely."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.models.versions import ModelVersion


@dataclass(frozen=True, slots=True)
class ModelFamilySpec:
    id: str
    display_name: str
    supported_tasks: frozenset[str]
    checkpoint_extensions: frozenset[str]
    adapter_name: str
    runnable: bool = False
    blocked_reason: str | None = None


FAMILIES = {
    "yolo11": ModelFamilySpec("yolo11", "YOLO11", frozenset({"detection2d"}), frozenset({".pt"}), "yolo11", True),
    "sam2": ModelFamilySpec(
        "sam2", "SAM2.1", frozenset({"segmentation"}), frozenset({".pt"}), "sam2", False, "WAITING_FOR_ARTIFACTS"
    ),
    "rtdetr": ModelFamilySpec(
        "rtdetr",
        "RT-DETR",
        frozenset({"detection2d"}),
        frozenset({".pt"}),
        "rtdetr",
        False,
        "BASE_CHECKPOINT_UNAVAILABLE",
    ),
    "faster_rcnn": ModelFamilySpec(
        "faster_rcnn",
        "Faster R-CNN",
        frozenset({"detection2d"}),
        frozenset({".pt", ".pth"}),
        "faster_rcnn",
        False,
        "BASE_CHECKPOINT_UNAVAILABLE",
    ),
    "centerpoint3d": ModelFamilySpec(
        "centerpoint3d",
        "CenterPoint",
        frozenset({"detection3d"}),
        frozenset({".pt", ".pth"}),
        "centerpoint3d",
        False,
        "WAITING_FOR_ARTIFACTS",
    ),
    "pointpillars3d": ModelFamilySpec(
        "pointpillars3d",
        "PointPillars",
        frozenset({"detection3d"}),
        frozenset({".pth"}),
        "pointpillars",
        False,
        "WAITING_FOR_GPU_VALIDATION",
    ),
}


def family_for_version(version: ModelVersion) -> ModelFamilySpec:
    model_name = version.model_name.lower()
    family_id = version.model_family_id or (
        "yolo11"
        if model_name.startswith("yolo")
        else "sam2"
        if model_name.startswith("sam")
        else "pointpillars3d"
        if model_name.startswith("pointpillars")
        else model_name
    )
    try:
        family = FAMILIES[family_id]
    except KeyError as exc:
        raise ValueError(f"MODEL_FAMILY_UNKNOWN: {version.model_name}") from exc
    if version.task not in family.supported_tasks:
        raise ValueError(f"MODEL_FAMILY_TASK_MISMATCH: {family.id} does not support {version.task}")
    return family


def adapter_request(
    version: ModelVersion, *, checkpoint: str, config: Any, settings: Settings
) -> tuple[str, dict[str, Any]]:
    """Return only constructor arguments owned by the selected model family."""
    family = family_for_version(version)
    if family.id == "yolo11":
        return family.adapter_name, {
            "weights": checkpoint,
            "device": settings.model_device,
            "batch_size": settings.model_batch_size,
            "half": settings.model_half_precision and settings.model_device.startswith("cuda"),
            "score_threshold": config.confidence_threshold,
        }
    if family.id == "sam2":
        sam_config = version.training_metadata.get("sam_config")
        if not sam_config:
            raise ValueError("MODEL_FAMILY_CONFIG_MISSING: SAM2 requires its model config")
        return family.adapter_name, {
            "weights": checkpoint,
            "config": str(sam_config),
            "device": settings.model_device,
            "mask_threshold": config.confidence_threshold,
        }
    if family.id == "pointpillars3d":
        model_config = version.training_metadata.get("model_config")
        if not model_config:
            raise ValueError("MODEL_FAMILY_CONFIG_MISSING: PointPillars requires MMDetection3D config")
        return family.adapter_name, {
            "weights": checkpoint,
            "config": str(model_config),
            "device": settings.model_device,
            "score_threshold": config.confidence_threshold,
        }
    raise ValueError(f"MODEL_FAMILY_NOT_RUNNABLE: {family.id}")
