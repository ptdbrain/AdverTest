"""Reviewed Cityscapes instance-segmentation source for the SAM2 protocol."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from src.core.types import Box, Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource
from src.datasets.io import load_image, load_mask

_LABELS = {24: "Person", 25: "Rider", 26: "Car", 27: "Truck", 28: "Bus", 32: "Motorcycle", 33: "Bicycle"}


class CityscapesParams(DatasetParams):
    root: str = "data/datasets/cityscapes"
    split: Literal["train", "val", "test"] = "train"
    anonymization_manifest: str = "manifest.jsonl"
    max_samples: int | None = Field(default=None, ge=1)


@DATASETS.register
class CityscapesSegmentationDataset(DatasetSource):
    name: ClassVar[str] = "cityscapes_segmentation"
    owner: ClassVar[str] = "group-c"
    loader_version: ClassVar[str] = "cityscapes-sam2-v1"
    params_model: ClassVar[type[DatasetParams]] = CityscapesParams
    task_id: ClassVar[str] = "segmentation"
    input_schema: ClassVar[tuple[str, ...]] = ("image",)
    annotation_schema: ClassVar[tuple[str, ...]] = ("instance_masks", "polygons", "class_labels")
    anonymized: ClassVar[bool] = True

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        raw_root = Path(self.params.root).expanduser()  # type: ignore[attr-defined]
        if not raw_root.is_absolute():
            from src.config import PROJECT_ROOT

            if (PROJECT_ROOT / raw_root).exists():
                self.root = (PROJECT_ROOT / raw_root).resolve()
            elif (PROJECT_ROOT.parent.parent / raw_root).exists():
                self.root = (PROJECT_ROOT.parent.parent / raw_root).resolve()
            else:
                self.root = raw_root.resolve()
        else:
            self.root = raw_root.resolve()
        self.manifest = self.root / self.params.anonymization_manifest  # type: ignore[attr-defined]
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Cityscapes SAM2 loader requires an anonymization manifest at {self.manifest}")

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name, anonymized=True, classes=tuple(_LABELS.values()), note="reviewed Cityscapes instance masks"
        )

    def load(self, limit: int | None = None) -> list[Sample]:
        return list(self.iter_samples(limit))

    def iter_samples(self, limit: int | None = None):
        """Yield samples lazily to keep full-resolution training bounded in RAM."""
        split = self.params.split  # type: ignore[attr-defined]
        image_root = self.root / "leftImg8bit" / split
        paths = sorted(image_root.rglob("*_leftImg8bit.png")) if image_root.exists() else []
        if not paths and (self.root / "leftImg8bit").exists():
            paths = sorted((self.root / "leftImg8bit").rglob("*_leftImg8bit.png"))
        if limit is None:
            limit = self.params.max_samples  # type: ignore[attr-defined]
        if limit is not None:
            paths = paths[:limit]
        for image_path in paths:
            stem = image_path.name.removesuffix("_leftImg8bit.png")
            try:
                rel = image_path.relative_to(self.root / "leftImg8bit")
                mask_path = self.root / "gtFine" / rel.parent / f"{stem}_gtFine_instanceIds.png"
            except ValueError:
                mask_path = self.root / "gtFine" / split / image_path.parent.name / f"{stem}_gtFine_instanceIds.png"
            if not mask_path.is_file():
                mask_path = self.root / "gtFine" / split / image_path.parent.name / f"{stem}_gtFine_instanceIds.png"
            if not mask_path.is_file():
                continue
            raw = load_mask(mask_path)
            assert raw is not None
            mask, labels, boxes, prompts = _instances(np.asarray(raw))
            if not labels:
                continue
            sample_id = f"{split}/{image_path.parent.name}/{stem}"
            yield Sample(
                sample_id=sample_id,
                image=load_image(image_path),
                boxes=boxes,
                mask=mask,
                anonymized=True,
                meta={
                    "instance_labels": labels,
                    "mask_reviewed": True,
                    "mask_source": "cityscapes_gtFine",
                    "sam_prompts": prompts,
                    "split": split,
                    "source_path": str(image_path),
                    "anonymization_manifest": str(self.manifest),
                },
            )


def _instances(raw: np.ndarray) -> tuple[np.ndarray, dict[int, str], tuple[Box, ...], list[dict[str, object]]]:
    compact = np.zeros(raw.shape, dtype=np.int32)
    labels: dict[int, str] = {}
    boxes: list[Box] = []
    prompts: list[dict[str, object]] = []
    next_id = 1
    for value in sorted(int(item) for item in np.unique(raw) if int(item) >= 1000):
        label_id = value // 1000
        label = _LABELS.get(label_id)
        if label is None:
            continue
        region = raw == value
        ys, xs = np.nonzero(region)
        if not len(xs):
            continue
        compact[region] = next_id
        x1, x2, y1, y2 = float(xs.min()), float(xs.max() + 1), float(ys.min()), float(ys.max() + 1)
        labels[next_id] = label
        boxes.append(Box(x1, y1, x2, y2, label))
        prompts.append({"prompt_id": f"gt-box:{next_id}", "object_id": next_id, "coordinates": [x1, y1, x2, y2]})
        next_id += 1
    return compact, labels, tuple(boxes), prompts
