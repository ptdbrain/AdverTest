"""Group C: random erasing / CutOut — rectangular parts of the frame are lost.

Severity controls the *total* erased area (plan §2 group C: 2 / 5 / 10 / 15 / 20 %
of the frame, spread over 1–3 regions). Everything about the layout — how many
regions, where they sit, how elongated they are — is drawn from ``ctx.rng``
*before* severity is applied, so the severity-5 mask grows out of the same
rectangles as the severity-1 mask. That makes the ladder nested (sanity check #2
of plan §3 holds structurally, not just on average) and makes the heatmap row
readable: same occluders, more of them covered.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field, model_validator

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.image_ops import FillMode, fill_values, paste
from src.core.types import AttackGroup, CostClass, Sample

#: (rows, cols) slices of one erased rectangle.
Region = tuple[slice, slice]


class RandomErasingParams(AttackParams):
    """Total erased area per severity, plus the shape of the individual holes."""

    area_fraction_per_severity: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20)
    min_regions: int = Field(default=1, ge=1, le=16)
    max_regions: int = Field(default=3, ge=1, le=16)
    aspect_range: tuple[float, float] = (0.3, 3.3)
    fill: FillMode = "noise"

    @model_validator(mode="after")
    def _check_ranges(self) -> RandomErasingParams:
        if self.min_regions > self.max_regions:
            raise ValueError("min_regions must not exceed max_regions")
        low, high = self.aspect_range
        if not 0.0 < low <= high:
            raise ValueError("aspect_range must satisfy 0 < low <= high")
        return self


@ATTACKS.register
class RandomErasing(BaseAttack):
    """Random rectangular regions erased (CutOut), area growing with severity."""

    name: ClassVar[str] = "random_erasing"
    group: ClassVar[AttackGroup] = "C"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "phong"
    reference: ClassVar[str] = (
        "Zhong et al., AAAI 2020 (arXiv:1708.04896); DeVries & Taylor (arXiv:1708.04552)"
    )
    params_model: ClassVar[type[AttackParams]] = RandomErasingParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: RandomErasingParams = self.params  # type: ignore[assignment]
        height, width = sample.image.shape[:2]
        budget = self.level(severity, params.area_fraction_per_severity) * height * width
        image = sample.image
        for region in self._layout(ctx.rng, height, width, budget):
            shape = (region[0].stop - region[0].start, region[1].stop - region[1].start, 3)
            image = paste(image, region, fill_values(params.fill, sample.image, shape, ctx.rng))
        return sample.with_image(image)

    def _layout(
        self,
        rng: np.random.Generator,
        height: int,
        width: int,
        budget: float,
    ) -> list[Region]:
        """Rectangles sharing ``budget`` pixels of area.

        The three ``rng`` draws happen in a fixed order and do not depend on the
        severity, so two severities of the same cell produce the same rectangle
        centres and aspect ratios — only their size changes.
        """
        params: RandomErasingParams = self.params  # type: ignore[assignment]
        count = int(rng.integers(params.min_regions, params.max_regions + 1))
        weights = rng.random(count) + 1e-3
        weights /= weights.sum()
        centres = rng.random((count, 2))
        aspects = rng.uniform(*params.aspect_range, size=count)
        regions: list[Region] = []
        for weight, (row, column), aspect in zip(weights, centres, aspects, strict=True):
            area = max(1.0, budget * float(weight))
            box_h = int(np.clip(round(float(np.sqrt(area / aspect))), 1, height))
            box_w = int(np.clip(round(float(np.sqrt(area * aspect))), 1, width))
            top = int(np.clip(round(float(row) * height - box_h / 2), 0, height - box_h))
            left = int(np.clip(round(float(column) * width - box_w / 2), 0, width - box_w))
            regions.append((slice(top, top + box_h), slice(left, left + box_w)))
        return regions
