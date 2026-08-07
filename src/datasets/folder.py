"""Folder-backed clean dataset in AdverTest or KITTI 2D format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from src.core.hashing import file_digest
from src.core.types import Box, Sample
from src.datasets import DATASETS
from src.datasets.base import (
    AnonymizationRequiredError,
    DatasetInfo,
    DatasetParams,
    DatasetSource,
)
from src.datasets.io import IMAGE_SUFFIXES, find_mask, load_boxes, load_image, load_mask

KITTI_LABEL_MAP = {
    "Car": "Car",
    "Van": "Car",
    "Truck": "Car",
    "Tram": "Car",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
    "Cyclist": "Cyclist",
}


class FolderDatasetParams(DatasetParams):
    root: str
    input_format: Literal["advertest", "kitti"] = "advertest"
    recursive: bool = False
    anonymization_manifest: str | None = None
    max_samples: int | None = Field(default=None, ge=1)


@DATASETS.register
class FolderDataset(DatasetSource):
    """Clean images and annotations loaded from a local folder."""

    name: ClassVar[str] = "folder_dataset"
    owner: ClassVar[str] = "core"
    loader_version: ClassVar[str] = "folder-v1"
    params_model: ClassVar[type[DatasetParams]] = FolderDatasetParams

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        typed: FolderDatasetParams = self.params  # type: ignore[assignment]
        self.root = Path(typed.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"folder dataset root does not exist: {self.root}")
        self._anonymized = self._read_anonymized()

    def _read_anonymized(self) -> bool:
        typed: FolderDatasetParams = self.params  # type: ignore[assignment]
        if typed.anonymization_manifest:
            return (self.root / typed.anonymization_manifest).is_file()
        descriptor = self.root / "dataset.json"
        if not descriptor.is_file():
            return False
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        return bool(payload.get("anonymized", False))

    def require_anonymized(self) -> None:
        if not self._anonymized:
            raise AnonymizationRequiredError(
                f"folder dataset {str(self.root)!r} is missing an anonymization manifest"
            )

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            anonymized=self._anonymized,
            note=f"{self.params.input_format} folder: {self.root}",  # type: ignore[attr-defined]
        )

    def load(self, limit: int | None = None) -> list[Sample]:
        typed: FolderDatasetParams = self.params  # type: ignore[assignment]
        effective = limit if limit is not None else typed.max_samples
        if typed.input_format == "kitti":
            return self._load_kitti(effective)
        return self._load_advertest(effective)

    def _load_advertest(self, limit: int | None) -> list[Sample]:
        typed: FolderDatasetParams = self.params  # type: ignore[assignment]
        images_root = self.root / "images"
        pattern = "**/*" if typed.recursive else "*"
        paths = sorted(
            path
            for path in images_root.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if limit is not None:
            paths = paths[:limit]
        samples: list[Sample] = []
        for path in paths:
            image = load_image(path)
            boxes = load_boxes(self.root / "labels" / f"{path.stem}.json")
            samples.append(
                Sample(
                    sample_id=path.stem,
                    image=image,
                    boxes=boxes,
                    mask=load_mask(find_mask(self.root / "masks", path.stem)),
                    depth=_load_depth(
                        self.root / "depths" / f"{path.stem}.npy",
                        image.shape[:2],
                    ),
                    anonymized=self._anonymized,
                    meta={
                        "source_path": str(path),
                        "source_uri": (
                            f"folder://advertest/{path.relative_to(images_root).as_posix()}"
                        ),
                        "source_format": "advertest",
                        "native_labels": tuple(box.label for box in boxes),
                        "loader_version": self.loader_version,
                        "split": self._dataset_split(),
                        "anonymization_manifest_hash": self._anonymization_manifest_hash(),
                    },
                )
            )
        return samples

    def _load_kitti(self, limit: int | None) -> list[Sample]:
        images_root = self.root / "image_2"
        paths = sorted(
            path
            for path in images_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if limit is not None:
            paths = paths[:limit]
        samples: list[Sample] = []
        for path in paths:
            label_path = self.root / "label_2" / f"{path.stem}.txt"
            boxes = _load_kitti_boxes(label_path)
            samples.append(
                Sample(
                    sample_id=path.stem,
                    image=load_image(path),
                    boxes=boxes,
                    anonymized=self._anonymized,
                    meta={
                        "source_path": str(path),
                        "source_uri": f"folder://kitti/{path.name}",
                        "source_format": "kitti",
                        "native_labels": _native_kitti_labels(label_path),
                        "loader_version": self.loader_version,
                        "split": self._dataset_split(),
                        "anonymization_manifest_hash": self._anonymization_manifest_hash(),
                    },
                )
            )
        return samples

    def _dataset_split(self) -> str:
        descriptor = self.root / "dataset.json"
        if not descriptor.is_file():
            return "unspecified"
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
        return str(payload.get("split", "unspecified"))

    def _anonymization_manifest_hash(self) -> str | None:
        typed: FolderDatasetParams = self.params  # type: ignore[assignment]
        manifest = (
            self.root / typed.anonymization_manifest
            if typed.anonymization_manifest
            else self.root / "dataset.json"
        )
        return file_digest(manifest, length=64) if manifest.is_file() else None


def _load_depth(path: Path, image_shape: tuple[int, int]) -> np.ndarray | None:
    """Load an optional metric/projection depth map for depth-aware weather."""
    if not path.is_file():
        return None
    depth = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if depth.shape != image_shape:
        raise ValueError(
            f"depth map {path} shape {depth.shape!r} does not match image {image_shape!r}"
        )
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError(f"depth map {path} must contain finite positive values")
    return np.ascontiguousarray(depth)


def _load_kitti_boxes(path: Path) -> tuple[Box, ...]:
    if not path.is_file():
        return ()
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        mapped = KITTI_LABEL_MAP.get(fields[0]) if len(fields) >= 8 else None
        if mapped is None:
            continue
        boxes.append(
            Box(
                x1=float(fields[4]),
                y1=float(fields[5]),
                x2=float(fields[6]),
                y2=float(fields[7]),
                label=mapped,
            )
        )
    return tuple(boxes)


def _native_kitti_labels(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    return tuple(
        fields[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if (fields := line.split())
    )
