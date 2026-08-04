"""Group B: deterministic depth-aware rain streaks and accumulation."""

from typing import ClassVar

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.attacks.weather._image import depth_for, replace_views, streak_layer, views
from src.core.image_ops import clip01
from src.core.types import AttackGroup, CostClass, SensorKind


class DepthRainParams(AttackParams):
    rate_per_severity: tuple[float, ...] = (10.0, 25.0, 50.0, 75.0, 100.0)
    streaks_per_megapixel_per_mmh: float = Field(default=0.35, gt=0.0)
    contrast_loss_per_mmh: float = Field(default=0.0015, gt=0.0, le=0.1)
    streak_length_per_severity: tuple[int, ...] = (3, 5, 7, 9, 12)


@ATTACKS.register
class DepthRain(BaseAttack):
    """Depth-aware rain streaks with contrast loss and windshield accumulation."""

    name: ClassVar[str] = "depth_rain"
    group: ClassVar[AttackGroup] = "B"
    cost_class: ClassVar[CostClass] = "cheap"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"image"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"image"})
    params_model: ClassVar[type[AttackParams]] = DepthRainParams

    def apply(self, sample, severity, ctx):
        params: DepthRainParams = self.params  # type: ignore[assignment]
        rate = self.level(severity, params.rate_per_severity)
        result = {}
        for view in views(sample):
            height, width = view.image.shape[:2]
            count = int(height * width / 1_000_000 * rate * params.streaks_per_megapixel_per_mmh)
            layer = streak_layer(ctx.rng, height, width, max(1, count), int(self.level(severity, params.streak_length_per_severity)))
            depth = depth_for(view.image, view.depth)
            visibility = np.exp(-params.contrast_loss_per_mmh * rate * depth / max(float(depth.mean()), 1.0))
            base = 0.5 + (view.image - 0.5) * visibility[..., None]
            result[view.name] = clip01(base + layer[..., None] * 0.35)
        return replace_views(sample, result)
