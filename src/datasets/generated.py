"""Reload a completed attack-dataset generation as a regular DatasetSource."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from src.core.hashing import array_digest, stable_digest
from src.core.types import Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource
from src.datasets.io import boxes_payload, load_boxes, load_image, load_mask


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
        self.records = _read_manifest(self.root / "manifest.jsonl")
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
            if stable_digest(boxes_payload(boxes), length=32) != record.get("label_hash"):
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
            samples.append(
                Sample(
                    sample_id=record["variant_id"],
                    image=image,
                    boxes=boxes,
                    mask=mask,
                    anonymized=bool(self.descriptor.get("anonymized", True)),
                    meta={"generation": record},
                )
            )
        return samples


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"generated dataset manifest not found: {path}")
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
