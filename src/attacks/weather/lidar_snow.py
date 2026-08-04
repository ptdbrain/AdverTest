"""Group B: LiDAR snowfall with range-dependent return loss."""

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.attacks.weather._lidar import snow_frame
from src.core.types import AttackGroup, CostClass, SensorKind


class LidarSnowParams(AttackParams):
    snowfall_rate_per_severity: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)


@ATTACKS.register
class LidarSnow(BaseAttack):
    """Snowfall simulation that removes and attenuates long-range returns."""

    name: ClassVar[str] = "lidar_snow"
    group: ClassVar[AttackGroup] = "B"
    modality: ClassVar[str] = "lidar"
    cost_class: ClassVar[CostClass] = "expensive"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    params_model: ClassVar[type[AttackParams]] = LidarSnowParams

    def apply(self, sample, severity, ctx):
        frame = sample.lidar_frame
        if frame is None:
            raise ValueError("lidar_snow requires sample.lidar_frame")
        return sample.with_lidar_frame(snow_frame(frame, self.level(severity, self.params.snowfall_rate_per_severity), ctx.rng))
