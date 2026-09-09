"""Group B: depth-aware snowflake overlay and atmospheric scattering."""

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackParams, BaseAttack
from src.attacks.weather._image import depth_for, replace_views, views
from src.core.image_ops import clip01
from src.core.types import AttackGroup, CostClass, SensorKind


class DepthSnowParams(AttackParams):
    flakes_per_megapixel: tuple[int, ...] = (200, 500, 1000, 2000, 4000)
    flake_radius_per_severity: tuple[int, ...] = (1, 1, 2, 2, 3)
    scattering_per_severity: tuple[float, ...] = (0.02, 0.04, 0.07, 0.11, 0.16)
    depth_falloff: float = Field(default=0.025, gt=0.0)
    depth_policy: Literal["required", "linear_prior"] = "linear_prior"


@ATTACKS.register
class DepthSnow(BaseAttack):
    """Depth-aware flakes with distance-dependent scattering."""

    name: ClassVar[str] = "depth_snow"
    group: ClassVar[AttackGroup] = "B"
    cost_class: ClassVar[CostClass] = "cheap"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"image"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"image"})
    owner: ClassVar[str] = "group-b"
    params_model: ClassVar[type[AttackParams]] = DepthSnowParams

    def apply(self, sample, severity, ctx):
        params: DepthSnowParams = self.params  # type: ignore[assignment]
        result = {}
        for view in views(sample):
            h, w = view.image.shape[:2]
            count = min(h * w, int(h * w / 1_000_000 * self.level(severity, params.flakes_per_megapixel)))
            mask = np.zeros((h, w), dtype=np.float32)
            if count:
                flat = ctx.rng.permutation(h * w)[:count]
                mask.flat[flat] = 1.0
            radius = int(self.level(severity, params.flake_radius_per_severity))
            for _ in range(radius):
                mask = np.maximum(mask, np.roll(mask, 1, axis=0))
                mask = np.maximum(mask, np.roll(mask, -1, axis=1))
            depth = depth_for(view.image, view.depth, policy=params.depth_policy)
            alpha = self.level(severity, params.scattering_per_severity)
            scatter = 1.0 - np.exp(-params.depth_falloff * depth)
            result[view.name] = clip01(view.image * (1 - alpha * scatter[..., None]) + mask[..., None] * 0.95)
        return replace_views(sample, result)
