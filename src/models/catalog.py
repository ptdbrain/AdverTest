"""Deployment-owned model catalog and its immutable GCS locations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import Settings
from src.models.versions import ModelVersion


class ModelVersionUnavailableError(LookupError):
    """A version is known but unsafe or impossible to execute."""


class ModelVersionCatalog:
    """Lookup boundary for immutable model versions."""

    def __init__(self, versions: Iterable[ModelVersion] = ()) -> None:
        self._versions = {version.id: version for version in versions}

    def list(self, *, task: str | None = None) -> list[ModelVersion]:
        versions = self._versions.values()
        if task is not None:
            versions = (version for version in versions if version.task == task)
        return sorted(versions, key=lambda version: version.id)

    def get(self, version_id: str) -> ModelVersion:
        try:
            return self._versions[version_id]
        except KeyError as exc:
            raise ModelVersionUnavailableError(f"MODEL_VERSION_UNKNOWN: {version_id}") from exc

    def require_runnable(self, version_id: str) -> ModelVersion:
        version = self.get(version_id)
        if not version.runnable:
            raise ModelVersionUnavailableError(version.blocked_reason or "MODEL_VERSION_NOT_RUNNABLE")
        return version


@dataclass(frozen=True, slots=True)
class CatalogModel:
    id: str
    model_name: str
    task: Literal["detection2d", "segmentation", "detection3d"]
    family_id: str
    storage_key: str
    filename: str
    runnable: bool
    blocked_reason: str | None = None
    metadata: dict[str, str] | None = None


CATALOG_MODELS: tuple[CatalogModel, ...] = (
    CatalogModel("yolo11s-base", "yolo11s", "detection2d", "yolo11", "catalog/models/detection2d/yolo11s/v1/yolo11s.pt", "yolo11s.pt", True),
    CatalogModel("rtdetr-l-base", "rtdetr-l", "detection2d", "rtdetr", "catalog/models/detection2d/rtdetr/v1/rtdetr-l.pt", "rtdetr-l.pt", True),
    CatalogModel("faster-rcnn-r50-base", "faster_rcnn", "detection2d", "faster_rcnn", "catalog/models/detection2d/faster_rcnn/v1/faster_rcnn.pth", "faster_rcnn.pth", True),
    CatalogModel("sam2-tiny-base", "sam2", "segmentation", "sam2", "catalog/models/segmentation/sam2_tiny/v1/sam2_hiera_t.pt", "sam2_hiera_t.pt", True, metadata={"config": "configs/sam2.1/sam2.1_hiera_t.yaml"}),
    CatalogModel("sam2-base-base", "sam2", "segmentation", "sam2", "catalog/models/segmentation/sam2_base/v1/sam2_hiera_b.pt", "sam2_hiera_b.pt", True, metadata={"config": "configs/sam2.1/sam2.1_hiera_b+.yaml"}),
    CatalogModel("mask-rcnn-r50-base", "mask_rcnn", "segmentation", "mask_rcnn", "catalog/models/segmentation/mask_rcnn/v1/mask_rcnn.pth", "mask_rcnn.pth", False, "ADAPTER_UNAVAILABLE"),
    CatalogModel("pointpillars-kitti-3class-base", "pointpillars-kitti-3class", "detection3d", "pointpillars3d", "catalog/models/detection3d/pointpillars/v1/pointpillars.pth", "pointpillars.pth", False, "WAITING_FOR_GPU_VALIDATION", metadata={"model_config": "configs/mmdet3d/pointpillars_hv_secfpn_6x8_160e_kitti-3d-3class.py"}),
    CatalogModel("bevfusion-base", "bevfusion", "detection3d", "bevfusion3d", "catalog/models/detection3d/bevfusion/v1/bevfusion.pth", "bevfusion.pth", False, "DATASET_MODALITY_MISMATCH"),
)


def catalog_model(model_id: str) -> CatalogModel | None:
    return next((item for item in CATALOG_MODELS if item.id == model_id), None)


def catalog_model_versions(*, platform: bool) -> list[ModelVersion]:
    """Expose server-owned catalog models when the worker is remote.

    A catalog entry is a planned artifact, not proof that its bytes are on the
    current worker.  Mark it runnable only after the fixed checkpoint target
    exists; otherwise the UI could select a phantom model and every attack
    would later fail with ``CHECKPOINT_MISSING``.
    """
    if not platform:
        return []
    versions: list[ModelVersion] = []
    for item in CATALOG_MODELS:
        checkpoint_path = Path("/app/data/checkpoints/catalog") / item.filename
        is_materialized = checkpoint_path.is_file()
        versions.append(ModelVersion(
            id=item.id,
            model_name=item.model_name,
            task=item.task,
            checkpoint_path=str(checkpoint_path),
            checkpoint_hash=None,
            parent_id=None,
            training_metadata={"source": "catalog", "storage_key": item.storage_key, **(item.metadata or {})},
            runnable=item.runnable and is_materialized,
            blocked_reason=item.blocked_reason or (None if is_materialized else "CHECKPOINT_MISSING"),
            model_family_id=item.family_id,
            checkpoint_role="base",
        ))
    return versions


def catalog_checkpoint_target(settings: Settings, model_id: str) -> Path | None:
    item = catalog_model(model_id)
    if item is None:
        return None
    return Path(settings.checkpoint_root).expanduser().resolve() / "catalog" / item.filename
