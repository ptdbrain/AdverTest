"""Reference dataset: deterministic synthetic scenes, no download required.

Bright rectangles on a dark textured background, with aspect ratios that encode
the three normalised classes and a forward-camera depth prior so depth-aware
weather attacks (group B) have something to work with.

``brightness_range`` is calibrated against ``blob_detector``: high enough that the
clean baseline is a clean AP 1.0, low enough that corruption severity ladders show
up as a graded AP curve instead of an all-or-nothing jump. Change it and the
degradation numbers of every demo shift with it.

Real datasets are separate files — see :mod:`src.datasets`.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from src.core.image_ops import clip01, linear_depth_prior
from src.core.types import Box, Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource

#: Aspect-ratio band per class; the reference detector reads the class back out of it.
CLASS_ASPECTS: dict[str, tuple[float, float]] = {
    "Car": (1.45, 2.10),
    "Pedestrian": (0.45, 0.70),
    "Cyclist": (0.88, 1.12),
}


class SyntheticShapesParams(DatasetParams):
    n_samples: int = Field(default=24, ge=1, le=2000)
    image_size: int = Field(default=96, ge=32, le=512)
    seed: int = 20260730
    min_objects: int = Field(default=2, ge=1)
    max_objects: int = Field(default=4, ge=1)
    brightness_range: tuple[float, float] = (0.68, 0.88)
    background_level: float = Field(default=0.12, ge=0.0, le=1.0)
    background_texture: float = Field(default=0.02, ge=0.0, le=0.5)
    depth_near_m: float = Field(default=5.0, gt=0.0)
    depth_far_m: float = Field(default=25.0, gt=0.0)


@DATASETS.register
class SyntheticShapes(DatasetSource):
    """Synthetic bright-rectangle scenes with 2D boxes and a depth prior."""

    name: ClassVar[str] = "synthetic_shapes"
    #: Pixels are generated, so there is no personal data to anonymise.
    anonymized: ClassVar[bool] = True
    owner: ClassVar[str] = "core"
    params_model: ClassVar[type[DatasetParams]] = SyntheticShapesParams

    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            anonymized=True,
            note="synthetic pixels — no personal data, anonymisation gate not applicable",
        )

    def load(self, limit: int | None = None) -> list[Sample]:
        params: SyntheticShapesParams = self.params  # type: ignore[assignment]
        count = params.n_samples if limit is None else min(limit, params.n_samples)
        return [self._make_sample(index) for index in range(count)]

    def _make_sample(self, index: int) -> Sample:
        params: SyntheticShapesParams = self.params  # type: ignore[assignment]
        rng = np.random.default_rng(params.seed + index)
        size = params.image_size
        image = self._background(rng, size)
        boxes: list[Box] = []
        for _ in range(int(rng.integers(params.min_objects, params.max_objects + 1))):
            placed = self._place_object(image, boxes, rng)
            if placed is not None:
                boxes.append(placed)
        depth = linear_depth_prior(size, size, near=params.depth_near_m, far=params.depth_far_m)
        mask = np.zeros((size, size), dtype=np.uint8)
        for object_index, box in enumerate(boxes, start=1):
            mask[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)] = object_index
        return Sample(
            sample_id=f"{self.name}_{index:04d}",
            image=clip01(image),
            boxes=tuple(boxes),
            mask=mask,
            depth=depth,
            anonymized=True,
            meta={"index": index},
        )

    def _background(self, rng: np.random.Generator, size: int) -> np.ndarray:
        params: SyntheticShapesParams = self.params  # type: ignore[assignment]
        texture = rng.normal(0.0, params.background_texture, size=(size, size, 3))
        return (params.background_level + texture).astype(np.float32)

    def _place_object(
        self,
        image: np.ndarray,
        existing: list[Box],
        rng: np.random.Generator,
    ) -> Box | None:
        """Draw one non-overlapping rectangle; returns ``None`` if no room was found."""
        params: SyntheticShapesParams = self.params  # type: ignore[assignment]
        size = params.image_size
        for _ in range(20):
            label = str(rng.choice(list(CLASS_ASPECTS)))
            box = self._sample_box(label, size, rng)
            if any(_overlaps(box, other) for other in existing):
                continue
            low, high = params.brightness_range
            brightness = float(rng.uniform(low, high))
            rows = slice(int(box.y1), int(box.y2))
            cols = slice(int(box.x1), int(box.x2))
            image[rows, cols, :] = brightness + rng.normal(0.0, 0.01, size=image[rows, cols, :].shape)
            return box
        return None

    @staticmethod
    def _sample_box(label: str, size: int, rng: np.random.Generator) -> Box:
        """Random box whose aspect ratio stays inside the class band."""
        low, high = CLASS_ASPECTS[label]
        aspect = float(rng.uniform(low, high))
        height = float(rng.integers(max(8, size // 8), max(9, size // 3)))
        width = float(np.clip(round(height * aspect), 6, size - 2))
        x1 = float(rng.integers(1, max(2, size - int(width) - 1)))
        y1 = float(rng.integers(1, max(2, size - int(height) - 1)))
        return Box(x1, y1, x1 + width, y1 + height, label, 1.0)


def _overlaps(box: Box, other: Box, *, margin: float = 3.0) -> bool:
    """True when two boxes touch, keeping a margin so components stay separable."""
    return not (
        box.x2 + margin <= other.x1
        or other.x2 + margin <= box.x1
        or box.y2 + margin <= other.y1
        or other.y2 + margin <= box.y1
    )
