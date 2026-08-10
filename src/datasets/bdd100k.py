"""BDD100K semantic-only external source; intentionally not SAM instance GT."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from src.core.types import Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource
from src.datasets.io import load_image, load_mask

_LABELS = {11: "Person", 12: "Rider", 13: "Car", 14: "Truck", 15: "Bus", 17: "Motorcycle", 18: "Bicycle"}


class BDD100KParams(DatasetParams):
    root: str
    split: Literal["train", "val", "test"] = "val"
    anonymization_manifest: str
    max_samples: int | None = Field(default=None, ge=1)


@DATASETS.register
class BDD100KSemanticDataset(DatasetSource):
    name: ClassVar[str] = "bdd100k_semantic"
    owner: ClassVar[str] = "group-c"
    loader_version: ClassVar[str] = "bdd100k-semantic-v1"
    params_model: ClassVar[type[DatasetParams]] = BDD100KParams

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.root = Path(self.params.root).expanduser().resolve()  # type: ignore[attr-defined]
        self.manifest = self.root / self.params.anonymization_manifest  # type: ignore[attr-defined]
        if not self.manifest.is_file():
            raise FileNotFoundError("BDD100K external loader requires an anonymization manifest")

    def info(self) -> DatasetInfo:
        return DatasetInfo(name=self.name, anonymized=True, classes=tuple(_LABELS.values()), note="external semantic-only; non-paired")

    def load(self, limit: int | None = None) -> list[Sample]:
        split = self.params.split  # type: ignore[attr-defined]
        image_paths = sorted((self.root / "10k" / split).glob("*.jpg"))
        if limit is None:
            limit = self.params.max_samples  # type: ignore[attr-defined]
        if limit is not None:
            image_paths = image_paths[:limit]
        if split == "test":
            return []
        samples: list[Sample] = []
        for image_path in image_paths:
            key = image_path.stem
            label_path = self.root / "labels" / split / f"{key}_{split}_id.png"
            semantic = load_mask(label_path)
            if semantic is None:
                continue
            binary = np.isin(semantic, tuple(_LABELS)).astype(np.uint8)
            samples.append(Sample(sample_id=f"{split}/{key}", image=load_image(image_path), mask=None, anonymized=True, meta={"semantic_mask": binary, "semantic_label_map": np.asarray(semantic), "semantic_classes": _LABELS, "comparison_scope": "external_semantic_nonpaired", "split": split, "source_path": str(image_path), "anonymization_manifest": str(self.manifest)}))
        return samples
