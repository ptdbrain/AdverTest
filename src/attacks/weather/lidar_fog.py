"""Group B: LiDAR fog with range attenuation and stochastic backscatter."""

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.base import AttackParams, BaseAttack
from src.attacks.weather._lidar import fog_frame
from src.core.types import AttackGroup, CostClass, SensorKind


class LidarFogParams(AttackParams):
    alpha_per_severity: tuple[float, ...] = (0.005, 0.01, 0.03, 0.06, 0.12)


@ATTACKS.register
class LidarFog(BaseAttack):
    """Range-dependent LiDAR attenuation and fog backscatter."""

    name: ClassVar[str] = "lidar_fog"
    group: ClassVar[AttackGroup] = "B"
    modality: ClassVar[str] = "lidar"
    cost_class: ClassVar[CostClass] = "medium"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    owner: ClassVar[str] = "group-b"
    params_model: ClassVar[type[AttackParams]] = LidarFogParams

    def apply(self, sample, severity, ctx):
        frame = sample.lidar_frame
        if frame is None:
            raise ValueError("lidar_fog requires sample.lidar_frame")
        return sample.with_lidar_frame(fog_frame(frame, self.level(severity, self.params.alpha_per_severity), ctx.rng))
