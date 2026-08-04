"""KITTI 2D source backed by an anonymized KITTI export."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from src.core.hashing import stable_digest
from src.core.types import Box, Modality, Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource
from src.datasets.io import IMAGE_SUFFIXES, load_image

LABEL_MAP = {
    "Car": "Car",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
}
VEHICLE_ALIASES = {"Van": "Car", "Truck": "Car"}
Difficulty = Literal["all", "easy", "moderate", "hard"]
DIFFICULTY_LIMITS: dict[Difficulty, tuple[float, int, float]] = {
    "easy": (40.0, 0, 0.15),
    "moderate": (25.0, 1, 0.30),
    "hard": (25.0, 2, 0.50),
    "all": (0.0, 3, 1.0),
}


class KittiParams(DatasetParams):
    """Selection and label policy for a KITTI export."""

    root: str = Field(
        default_factory=lambda: os.environ.get(
            "ADVERTEST_KITTI_ROOT", "data/anonymized/kitti"
        )
    )
    split: Literal["train", "val", "all"] = "val"
    difficulty: Difficulty = "moderate"
    anonymize: Literal["required", "off"] = "required"
    merge_van_truck: bool = False
    sample_ids: tuple[str, ...] | None = None
    manifest_path: str | None = None


@DATASETS.register
class Kitti(DatasetSource):
    """KITTI 2D detection with a strict anonymization manifest gate."""

    name: ClassVar[str] = "kitti"
    anonymized: ClassVar[bool] = False
    modality: ClassVar[Modality] = "image"
    owner: ClassVar[str] = "core"
    params_model: ClassVar[type[DatasetParams]] = KittiParams

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        settings: KittiParams = self.params  # type: ignore[assignment]
        self.root = Path(settings.root).expanduser().resolve()
        self.image_dir = self._find_dir("image_2")
        self.label_dir = self._find_dir("label_2")
        self.anonymized = settings.anonymize == "required" and self._has_manifest()

    def _find_dir(self, name: str) -> Path:
        direct = self.root / name
        nested = self.root / "training" / name
        return direct if direct.is_dir() else nested

    def _has_manifest(self) -> bool:
        settings: KittiParams = self.params  # type: ignore[assignment]
        descriptor = self.root / "dataset.json"
        if not descriptor.is_file():
            return False
        try:
            payload = json.loads(descriptor.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        manifest = (
            Path(settings.manifest_path).expanduser()
            if settings.manifest_path
            else self.root / "manifest.jsonl"
        )
        return bool(payload.get("anonymized", False)) and manifest.is_file()

    def info(self) -> DatasetInfo:
        settings: KittiParams = self.params  # type: ignore[assignment]
        return DatasetInfo(
            name=self.name,
            anonymized=self.anonymized,
            modality=self.modality,
            note=(
                f"root={self.root} split={settings.split} "
                f"difficulty={settings.difficulty}"
            ),
        )

    def load(self, limit: int | None = None) -> list[Sample]:
        self._require_layout()
        settings: KittiParams = self.params  # type: ignore[assignment]
        ids = list(self._ids())
        if settings.sample_ids is not None:
            missing = sorted(set(settings.sample_ids) - set(ids))
            if missing:
                raise ValueError(f"KITTI sample_ids do not exist: {missing}")
            ids = list(settings.sample_ids)
        if limit is not None:
            ids = ids[:limit]
        return [self._load_sample(image_id) for image_id in ids]

    def _require_layout(self) -> None:
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                f"KITTI image_2/label_2 directories do not exist under {self.root}"
            )

    def _ids(self) -> Iterable[str]:
        settings: KittiParams = self.params  # type: ignore[assignment]
        if settings.sample_ids is not None:
            return settings.sample_ids
        split_file = self.root / "ImageSets" / f"{settings.split}.txt"
        if split_file.is_file():
            return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        return sorted(
            path.stem
            for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def _load_sample(self, image_id: str) -> Sample:
        settings: KittiParams = self.params  # type: ignore[assignment]
        image_path = self.image_dir / f"{image_id}.png"
        if not image_path.is_file():
            candidates = [
                path for path in self.image_dir.glob(f"{image_id}.*") if path.is_file()
            ]
            if not candidates:
                raise FileNotFoundError(f"KITTI image not found: {image_id}")
            image_path = candidates[0]
        image = load_image(image_path)
        boxes, dropped = self._read_labels(
            self.label_dir / f"{image_id}.txt", image.shape[:2]
        )
        variant = stable_digest(
            {
                "root": str(self.root),
                "anonymize": settings.anonymize,
                "difficulty": settings.difficulty,
                "merge_van_truck": settings.merge_van_truck,
            },
            length=8,
        )
        return Sample(
            sample_id=f"kitti_{variant}_{image_id}",
            image=image,
            boxes=boxes,
            anonymized=self.anonymized,
            meta={
                "image_id": image_id,
                "source_path": str(image_path),
                "source_format": "kitti",
                "dropped_labels": dropped,
            },
        )

    def _read_labels(
        self, path: Path, shape: tuple[int, int]
    ) -> tuple[tuple[Box, ...], dict[str, int]]:
        if not path.is_file():
            raise FileNotFoundError(f"KITTI label file not found: {path}")
        height, width = shape
        settings: KittiParams = self.params  # type: ignore[assignment]
        boxes: list[Box] = []
        dropped: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 8:
                continue
            box = self._parse_row(fields, width, height, settings, dropped)
            if box is not None:
                boxes.append(box)
        return tuple(boxes), dropped

    @staticmethod
    def _parse_row(
        fields: Sequence[str],
        width: int,
        height: int,
        settings: KittiParams,
        dropped: dict[str, int],
    ) -> Box | None:
        raw_label = fields[0]
        label = LABEL_MAP.get(raw_label)
        if label is None and settings.merge_van_truck:
            label = VEHICLE_ALIASES.get(raw_label)
        if label is None:
            dropped[raw_label] = dropped.get(raw_label, 0) + 1
            return None
        truncated = float(fields[1])
        occluded = int(float(fields[2]))
        x1, y1, x2, y2 = (float(value) for value in fields[4:8])
        x1, x2 = np.clip([x1, x2], 0.0, float(width))
        y1, y2 = np.clip([y1, y2], 0.0, float(height))
        min_height, max_occlusion, max_truncation = DIFFICULTY_LIMITS[
            settings.difficulty
        ]
        if x2 <= x1 or y2 <= y1:
            dropped["degenerate"] = dropped.get("degenerate", 0) + 1
            return None
        if (
            (y2 - y1) < min_height
            or occluded > max_occlusion
            or truncated > max_truncation
        ):
            key = f"{raw_label}:difficulty"
            dropped[key] = dropped.get(key, 0) + 1
            return None
        return Box(float(x1), float(y1), float(x2), float(y2), label, 1.0)
