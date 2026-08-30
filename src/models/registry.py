"""Model Version Registry and Checkpoint Manager for AdverTest."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.adapters import get_adapter
from src.adapters.base import ModelAdapter
from src.core.hashing import file_digest
from src.core.types import Task


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Registered Model Version metadata and lineage info."""

    version_id: str
    model_id: str
    adapter_name: str
    task: Task
    checkpoint_path: str
    parent_version_id: str | None = None
    description: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoint_hash(self) -> str | None:
        path = Path(self.checkpoint_path).expanduser()
        return file_digest(path) if path.is_file() else None

    @property
    def is_available(self) -> bool:
        path = Path(self.checkpoint_path).expanduser()
        return path.is_file()

    def as_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "adapter_name": self.adapter_name,
            "task": self.task,
            "checkpoint_path": self.checkpoint_path,
            "parent_version_id": self.parent_version_id,
            "description": self.description,
            "checkpoint_hash": self.checkpoint_hash,
            "is_available": self.is_available,
            "metrics": self.metrics,
        }


# Model Catalog Registry
REGISTERED_MODEL_VERSIONS: dict[str, ModelVersion] = {
    # Person B (YOLO11s Detection)
    "yolo11s-clean-b0": ModelVersion(
        version_id="yolo11s-clean-b0",
        model_id="yolo11s",
        adapter_name="yolo11",
        task="detection2d",
        checkpoint_path="runs/train/yolo_b0/yolo11s-clean-b0_best.pt",
        parent_version_id=None,
        description="YOLO11s baseline clean model trained on KITTI split v1",
        metrics={},
    ),
    "yolo11s-robust-r1": ModelVersion(
        version_id="yolo11s-robust-r1",
        model_id="yolo11s",
        adapter_name="yolo11",
        task="detection2d",
        checkpoint_path="runs/train/yolo_r1/yolo11s-robust-r1_best.pt",
        parent_version_id="yolo11s-clean-b0",
        description="YOLO11s robust mix fine-tuned on combined corruptions & adversarial data",
        metrics={},
    ),
    "yolo11s-repaired-r2-fog": ModelVersion(
        version_id="yolo11s-repaired-r2-fog",
        model_id="yolo11s",
        adapter_name="yolo11",
        task="detection2d",
        checkpoint_path="runs/train/yolo_r2_fog/yolo11s-repaired-r2-fog_best.pt",
        parent_version_id="yolo11s-robust-r1",
        description="YOLO11s targeted repair model for Fog corruption recovery",
        metrics={},
    ),
    "yolo11s-repaired-r2-sensor": ModelVersion(
        version_id="yolo11s-repaired-r2-sensor",
        model_id="yolo11s",
        adapter_name="yolo11",
        task="detection2d",
        checkpoint_path="runs/train/yolo_r2_sensor/yolo11s-repaired-r2-sensor_fault_best.pt",
        parent_version_id="yolo11s-robust-r1",
        description="YOLO11s targeted repair model for Sensor Fault recovery",
        metrics={},
    ),
    # Person C (SAM2.1 Segmentation)
    "sam21-clean-b0": ModelVersion(
        version_id="sam21-clean-b0",
        model_id="sam21",
        adapter_name="sam2",
        task="segmentation",
        checkpoint_path="runs/train/sam2_b0/sam21-clean-b0_best.pt",
        parent_version_id=None,
        description="SAM2.1 Hiera Small baseline clean segmentation model",
        metrics={},
    ),
    "sam21-robust-r1": ModelVersion(
        version_id="sam21-robust-r1",
        model_id="sam21",
        adapter_name="sam2",
        task="segmentation",
        checkpoint_path="runs/train/sam2_r1/sam21-robust-r1_best.pt",
        parent_version_id="sam21-clean-b0",
        description="SAM2.1 Hiera Small robust fine-tuned segmentation model",
        metrics={},
    ),
}


def list_model_versions(task: Task | None = None) -> list[ModelVersion]:
    """List all registered model versions, optionally filtered by task."""
    versions = list(REGISTERED_MODEL_VERSIONS.values())
    if task is not None:
        versions = [v for v in versions if v.task == task]
    return versions


def get_model_version(version_id: str) -> ModelVersion:
    """Retrieve ModelVersion by version_id."""
    if version_id not in REGISTERED_MODEL_VERSIONS:
        available = ", ".join(REGISTERED_MODEL_VERSIONS.keys())
        raise KeyError(f"Unknown model version_id {version_id!r}. Available: {available}")
    return REGISTERED_MODEL_VERSIONS[version_id]


def get_adapter_for_version(version_id: str, **kwargs: object) -> ModelAdapter:
    """Instantiate a ModelAdapter configured with the checkpoint of the model version."""
    model_version = get_model_version(version_id)
    return get_adapter(model_version.adapter_name, weights=model_version.checkpoint_path, **kwargs)
