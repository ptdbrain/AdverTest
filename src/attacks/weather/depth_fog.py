"""Group B: depth-aware atmospheric-scattering fog for camera views."""

from typing import ClassVar

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.attacks.weather._image import depth_for, fog, replace_views, views
from src.core.types import AttackGroup, CostClass, SensorKind


class DepthFogParams(AttackParams):
    beta_per_severity: tuple[float, ...] = (0.03, 0.06, 0.10, 0.15, 0.20)
    airlight: float = Field(default=1.0, ge=0.0, le=1.0)


@ATTACKS.register
class DepthFog(BaseAttack):
    """Depth-aware atmospheric scattering using projected depth or a prior."""

    name: ClassVar[str] = "depth_fog"
    group: ClassVar[AttackGroup] = "B"
    cost_class: ClassVar[CostClass] = "cheap"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"image"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"image"})
    params_model: ClassVar[type[AttackParams]] = DepthFogParams

    def apply(self, sample, severity, ctx):
        beta = self.level(severity, self.params.beta_per_severity)
        result = {
            view.name: fog(view.image, depth_for(view.image, view.depth), beta, self.params.airlight)
            for view in views(sample)
        }
        return replace_views(sample, result)
