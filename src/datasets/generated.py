"""Reload a completed attack-dataset generation as a regular DatasetSource."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.core.hashing import array_digest, stable_digest
from src.core.types import CameraView, LidarFrame, Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource
from src.datasets.io import (
    annotations_payload,
    boxes_payload,
    load_boxes,
    load_boxes3d,
    load_image,
    load_mask,
    migrate_generated_record,
)


class GeneratedDatasetParams(DatasetParams):
    root: str


@DATASETS.register
class GeneratedDatasetSource(DatasetSource):
    """Generated variants described by an AdverTest manifest."""

    name: ClassVar[str] = "generated_dataset"
    anonymized: ClassVar[bool] = True
    owner: ClassVar[str] = "core"
    params_model: ClassVar[type[DatasetParams]] = GeneratedDatasetParams

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.root = Path(self.params.root).expanduser().resolve()  # type: ignore[attr-defined]
        descriptor = self.root / "dataset.json"
        if not descriptor.is_file():
            raise FileNotFoundError(f"generated dataset descriptor not found: {descriptor}")
        self.descriptor = json.loads(descriptor.read_text(encoding="utf-8"))
        if self.descriptor.get("status") != "complete":
            raise ValueError(f"generated dataset is not complete: {self.root}")
        self.records = _read_manifest(
            self.root / "manifest.jsonl",
            dataset_format=str(self.descriptor.get("format", "")),
        )
        if self.descriptor.get("n_variants") != len(self.records):
            raise ValueError("generated dataset manifest count does not match dataset.json")
        manifest_hash = stable_digest(
            sorted(self.records, key=lambda row: row["variant_id"]),
            length=32,
        )
        if self.descriptor.get("manifest_hash") != manifest_hash:
            raise ValueError("generated dataset manifest hash does not match dataset.json")

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            anonymized=bool(self.descriptor.get("anonymized", True)),
            note=f"generated attack dataset: {self.root}",
        )

    def load(self, limit: int | None = None) -> list[Sample]:
        records = self.records
        if limit is not None:
            records = records[:limit]
        samples: list[Sample] = []
        for record in records:
            image = load_image(self.root / record["image_path"])
            if array_digest(image, length=32) != record.get("output_hash"):
                raise ValueError(
                    f"generated image hash mismatch for {record['variant_id']!r}"
                )
            boxes = load_boxes(self.root / record["label_path"])
            boxes3d = load_boxes3d(self.root / record["label_path"])
            label_payload = (
                annotations_payload(boxes, boxes3d)
                if record.get("annotation_format") == "advertest-annotations-v2"
                else boxes_payload(boxes)
            )
            if stable_digest(label_payload, length=32) != record.get("label_hash"):
                raise ValueError(
                    f"generated label hash mismatch for {record['variant_id']!r}"
                )
            mask = load_mask(
                self.root / record["mask_path"] if record.get("mask_path") else None
            )
            expected_mask_hash = record.get("mask_hash")
            actual_mask_hash = (
                array_digest(mask, length=32)
                if mask is not None
                else None
            )
            if actual_mask_hash != expected_mask_hash:
                raise ValueError(
                    f"generated mask hash mismatch for {record['variant_id']!r}"
                )
            cameras = tuple(
                _load_camera(self.root, payload)
                for payload in record.get("camera_payloads", [])
            )
            if not cameras:
                cameras = tuple(
                    CameraView(name, _load_hashed_array(self.root, path, None))
                    for name, path in record.get("camera_paths", {}).items()
                )
            lidar = _load_lidar(self.root, record)
            samples.append(
                Sample(
                    sample_id=record["variant_id"],
                    image=image,
                    boxes=boxes,
                    boxes3d=boxes3d,
                    mask=mask,
                    camera_views=cameras,
                    lidar_frame=lidar,
                    anonymized=bool(self.descriptor.get("anonymized", True)),
                    meta={"generation": record},
                )
            )
        return samples


def _read_manifest(
    path: Path,
    *,
    dataset_format: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"generated dataset manifest not found: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(
                migrate_generated_record(
                    json.loads(line),
                    dataset_format=dataset_format,
                )
            )
    return records


def _load_camera(root: Path, payload: dict[str, Any]) -> CameraView:
    return CameraView(
        name=str(payload["name"]),
        image=_load_hashed_array(root, payload.get("image_path"), payload.get("image_hash")),
        depth=_load_hashed_array(root, payload.get("depth_path"), payload.get("depth_hash")),
        intrinsic=_load_hashed_array(root, payload.get("intrinsic_path"), payload.get("intrinsic_hash")),
        sensor_to_ego=_load_hashed_array(root, payload.get("sensor_to_ego_path"), payload.get("sensor_to_ego_hash")),
        previous_image=_load_hashed_array(root, payload.get("previous_image_path"), payload.get("previous_image_hash")),
    )


def _load_lidar(root: Path, record: dict[str, Any]) -> LidarFrame | None:
    path = record.get("lidar_path")
    if path is None:
        return None
    points = _load_hashed_array(root, path, record.get("lidar_hash"))
    if points is None:
        return None
    fields = tuple(record.get("lidar_fields") or ("x", "y", "z", "intensity", "ring"))
    return LidarFrame(
        points=np.asarray(points, dtype=np.float32),
        fields=fields,
        sensor_model=str(record.get("lidar_sensor_model") or "unknown"),
    )


def _load_hashed_array(root: Path, relative: str | None, expected_hash: str | None) -> np.ndarray | None:
    if relative is None:
        return None
    array = np.load(root / relative, allow_pickle=False)
    if expected_hash is not None and array_digest(array, length=32) != expected_hash:
        raise ValueError(f"generated array hash mismatch: {relative}")
    return np.ascontiguousarray(array)
