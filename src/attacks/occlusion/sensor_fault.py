"""Group C: camera hardware faults — stuck pixels plus lost readout bands.

Plan §2 group C lists "camera dropout" for the 6-camera nuScenes rig. KITTI and
every other single-camera dataset cannot drop *a* camera, so the same failure
mode is modelled where it is observable on one sensor: dead pixels from CMOS
defects, and horizontal bands lost to a readout/transport glitch. Severity raises
both at once.

Both are nested across severity. The dead-pixel mask is the ``n`` smallest values
of a single uniform field drawn once per cell, so the severity-5 mask is a strict
superset of the severity-1 mask; the bands are the first ``k`` of a fixed list of
positions. At least one pixel always dies, so severity 1 is never a silent no-op
on a small frame.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.image_ops import FillMode, fill_values, paste
from src.core.types import AttackGroup, CostClass, Sample

Region = tuple[slice, slice]


class SensorFaultParams(AttackParams):
    """Dead-pixel fraction and number of lost readout bands, per severity."""

    dead_pixel_fraction_per_severity: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05)
    n_bands_per_severity: tuple[int, ...] = (0, 1, 2, 3, 4)
    band_height_px: int = Field(default=8, ge=1)
    fill: FillMode = "black"


@ATTACKS.register
class SensorFault(BaseAttack):
    """Stuck pixels and lost readout bands from a faulty camera sensor."""

    name: ClassVar[str] = "sensor_fault"
    group: ClassVar[AttackGroup] = "C"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "phong"
    reference: ClassVar[str] = "AdverTest plan §2 group C (camera dropout, single-sensor analogue)"
    params_model: ClassVar[type[AttackParams]] = SensorFaultParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: SensorFaultParams = self.params  # type: ignore[assignment]
        height, width = sample.image.shape[:2]
        # Both draws happen before severity is read, which is what makes the
        # severity ladder nested rather than merely "bigger on average".
        field = ctx.rng.random((height, width))
        band_offsets = ctx.rng.random(max(params.n_bands_per_severity, default=0))

        image = sample.image.copy()
        dead = self._dead_mask(field, self.level(severity, params.dead_pixel_fraction_per_severity))
        image[dead] = fill_values(params.fill, sample.image, (int(dead.sum()), 3), ctx.rng)

        n_bands = int(self.level(severity, params.n_bands_per_severity))
        for region in self._bands(band_offsets[:n_bands], height, width):
            shape = (region[0].stop - region[0].start, width, 3)
            image = paste(image, region, fill_values(params.fill, sample.image, shape, ctx.rng))
        return sample.with_image(image)

    @staticmethod
    def _dead_mask(field: np.ndarray, fraction: float) -> np.ndarray:
        """The ``n`` lowest-valued pixels of ``field`` — nested in ``fraction``."""
        n_dead = int(np.clip(round(fraction * field.size), 1, field.size))
        threshold = float(np.partition(field.ravel(), n_dead - 1)[n_dead - 1])
        return field <= threshold

    def _bands(self, offsets: np.ndarray, height: int, width: int) -> list[Region]:
        """Horizontal stripes of lost scanlines at fixed random positions."""
        params: SensorFaultParams = self.params  # type: ignore[assignment]
        band = int(np.clip(params.band_height_px, 1, height))
        regions: list[Region] = []
        for offset in offsets:
            top = int(np.clip(round(float(offset) * (height - band)), 0, height - band))
            regions.append((slice(top, top + band), slice(0, width)))
        return regions
