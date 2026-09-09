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
    # Native output classes of the reference weights for this family. Used by
    # the UI mapping matrix when a checkpoint record has no platform metadata.
    default_class_names: tuple[str, ...] = ()
    # Code-owned reference adapters have no model weight to scan. Every
    # catalog/upload family remains checkpoint-backed by default.
    requires_checkpoint: bool = True


COCO_DETECTION_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

KITTI_3D_CLASSES = ("Car", "Pedestrian", "Cyclist")

FAMILIES = {
    "blob_detector": ModelFamilySpec(
        "blob_detector",
        "Blob Detector (portable CPU reference)",
        frozenset({"detection2d"}),
        frozenset(),
        "blob_detector",
        True,
        default_class_names=KITTI_3D_CLASSES,
        requires_checkpoint=False,
    ),
    "yolo11": ModelFamilySpec(
        "yolo11", "YOLO11", frozenset({"detection2d"}), frozenset({".pt"}), "yolo11", True,
        default_class_names=COCO_DETECTION_CLASSES,
    ),
    "sam2": ModelFamilySpec(
        # SAM2 is promptable and class-agnostic: its mapping matrix is driven
        # by dataset classes, so no native class list applies.
        "sam2", "SAM2.1", frozenset({"segmentation"}), frozenset({".pt"}), "sam2", False, "WAITING_FOR_ARTIFACTS",
    ),
    "rtdetr": ModelFamilySpec(
        "rtdetr",
        "RT-DETR",
        frozenset({"detection2d"}),
        frozenset({".pt"}),
        "rtdetr",
        True,
        default_class_names=COCO_DETECTION_CLASSES,
    ),
    "faster_rcnn": ModelFamilySpec(
        "faster_rcnn",
        "Faster R-CNN",
        frozenset({"detection2d"}),
        frozenset({".pt", ".pth"}),
        "faster_rcnn",
        True,
        default_class_names=COCO_DETECTION_CLASSES,
    ),
    "centerpoint3d": ModelFamilySpec(
        "centerpoint3d",
        "CenterPoint",
        frozenset({"detection3d"}),
        frozenset({".pt", ".pth"}),
        "centerpoint3d",
        False,
        "WAITING_FOR_ARTIFACTS",
        default_class_names=KITTI_3D_CLASSES,
    ),
    "pointpillars3d": ModelFamilySpec(
        "pointpillars3d",
        "PointPillars",
        frozenset({"detection3d"}),
        frozenset({".pth"}),
        "pointpillars",
        False,
        "WAITING_FOR_GPU_VALIDATION",
        default_class_names=KITTI_3D_CLASSES,
    ),
}

# Uploads name a deployment-owned config reference; callers never submit an
# arbitrary filesystem path that a later GPU worker could load.
APPROVED_MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "pointpillars3d": {
        "pointpillars-kitti-3class": "configs/mmdet3d/pointpillars_hv_secfpn_6x8_160e_kitti-3d-3class.py",
    },
}


def approved_model_config(family_id: str, config_id: str) -> str | None:
    """Return a deployment-approved model config reference for an upload."""
    return APPROVED_MODEL_CONFIGS.get(family_id, {}).get(config_id)


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
    if family.id == "blob_detector":
        return family.adapter_name, {}
    if family.id == "yolo11":
        return family.adapter_name, {
            "weights": checkpoint,
            "device": settings.model_device,
            "batch_size": settings.model_batch_size,
            "half": settings.model_half_precision and settings.model_device.startswith("cuda"),
            "score_threshold": config.confidence_threshold,
        }
    if family.id == "sam2":
        sam_config = version.training_metadata.get("sam_config") or version.training_metadata.get("config")
        if not sam_config:
            raise ValueError("MODEL_FAMILY_CONFIG_MISSING: SAM 2 requires a YAML model configuration")
        return family.adapter_name, {
            "weights": checkpoint,
            "config": str(sam_config),
            "device": settings.model_device,
            "mask_threshold": config.confidence_threshold,
        }
    if family.id == "rtdetr":
        return family.adapter_name, {
            "weights": checkpoint,
            "device": settings.model_device,
            "batch_size": settings.model_batch_size,
            "half": settings.model_half_precision and settings.model_device.startswith("cuda"),
            "score_threshold": config.confidence_threshold,
        }
    if family.id == "faster_rcnn":
        return family.adapter_name, {
            "weights": checkpoint,
            "device": settings.model_device,
            "score_threshold": config.confidence_threshold,
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
