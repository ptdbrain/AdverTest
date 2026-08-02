"""Group C: a distractor is pasted over each ground-truth object.

Unlike :mod:`~src.attacks.occlusion.random_erasing`, which hits the frame blindly,
this one is aimed: severity is the fraction ``r`` of every GT box that gets
covered, so the heatmap row answers "how much of a car may be hidden before the
detector drops it?".

Two deliberate choices worth knowing before reading the numbers:

* Plan §2 group C lists four ratios (10 / 25 / 50 / 75 %). The catalog keeps the
  standard five severity levels and adds 90 % as level 5 — a four-level attack
  would raise in :meth:`BaseAttack.run` for the default ``severities = [1, 3, 5]``
  and take the whole run down with it.
* On a frame without ground truth the attack would silently be a no-op, so it
  falls back to a centred region covering the same fraction of the whole frame.
  Group C is still measured on those frames, just without the aiming.

The occluder is a band, not a scatter of pixels: real occluders (another car, a
pole, a pedestrian in front) remove a contiguous part of the silhouette. It grows
from the anchor edge, so severity ladders are nested.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.image_ops import FillMode, box_slice, fill_values, paste
from src.core.types import AttackGroup, Box, CostClass, Sample

Anchor = Literal["bottom", "center", "top", "random"]
Region = tuple[slice, slice]


class ObjectOcclusionParams(AttackParams):
    """Covered fraction of each GT box per severity, and how the band is placed."""

    cover_ratio_per_severity: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    fill: FillMode = "mean"
    anchor: Anchor = "bottom"
    #: Occlude only the first N boxes of a frame; ``None`` means every box.
    max_objects: int | None = None


@ATTACKS.register
class ObjectOcclusion(BaseAttack):
    """Distractor pasted over each GT box, covering 10-90 % of it by severity."""

    name: ClassVar[str] = "object_occlusion"
    group: ClassVar[AttackGroup] = "C"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "phong"
    reference: ClassVar[str] = (
        "AdverTest plan §2 group C; occlusion sensitivity: Zeiler & Fergus, ECCV 2014 (arXiv:1311.2901)"
    )
    params_model: ClassVar[type[AttackParams]] = ObjectOcclusionParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: ObjectOcclusionParams = self.params  # type: ignore[assignment]
        ratio = self.level(severity, params.cover_ratio_per_severity)
        height, width = sample.image.shape[:2]
        # Drawn once, before severity is used, so the ladder stays nested.
        offset = float(ctx.rng.random())
        image = sample.image
        for region in self._regions(sample, height, width, ratio, offset):
            shape = (region[0].stop - region[0].start, region[1].stop - region[1].start, 3)
            image = paste(image, region, fill_values(params.fill, sample.image, shape, ctx.rng))
        return sample.with_image(image)

    def _regions(
        self,
        sample: Sample,
        height: int,
        width: int,
        ratio: float,
        offset: float,
    ) -> list[Region]:
        """One occluding band per targeted box, or the whole-frame fallback."""
        targets = self._targets(sample.boxes)
        if not targets:
            return [self._band(slice(0, height), slice(0, width), ratio, offset)]
        regions: list[Region] = []
        for box in targets:
            clipped = box_slice(box, height, width)
            if clipped is not None:
                regions.append(self._band(clipped[0], clipped[1], ratio, offset))
        return regions

    def _targets(self, boxes: tuple[Box, ...]) -> tuple[Box, ...]:
        params: ObjectOcclusionParams = self.params  # type: ignore[assignment]
        if params.max_objects is None:
            return boxes
        return boxes[: max(0, params.max_objects)]

    def _band(self, rows: slice, columns: slice, ratio: float, offset: float) -> Region:
        """Horizontal band covering ``ratio`` of the region, placed by ``anchor``."""
        params: ObjectOcclusionParams = self.params  # type: ignore[assignment]
        span = rows.stop - rows.start
        band = int(np.clip(round(span * ratio), 1, span))
        free = span - band
        if params.anchor == "bottom":
            top = rows.stop - band
        elif params.anchor == "top":
            top = rows.start
        elif params.anchor == "center":
            top = rows.start + free // 2
        else:
            top = rows.start + int(round(offset * free))
        return slice(top, top + band), columns
