"""Group C: uniform random LiDAR point dropout."""

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.base import AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, LidarFrame, SensorKind


class LidarPointDropoutParams(AttackParams):
    fraction_per_severity: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50)


@ATTACKS.register
class LidarPointDropout(BaseAttack):
    name: ClassVar[str] = "lidar_point_dropout"
    group: ClassVar[AttackGroup] = "C"
    modality: ClassVar[str] = "lidar"
    cost_class: ClassVar[CostClass] = "cheap"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    owner: ClassVar[str] = "3d-evaluation"
    params_model: ClassVar[type[AttackParams]] = LidarPointDropoutParams

    def apply(self, sample, severity, ctx):
        frame = sample.lidar_frame
        if frame is None:
            raise ValueError("lidar_point_dropout requires sample.lidar_frame")
        drop_fraction = self.level(severity, self.params.fraction_per_severity)
        count = len(frame.points)
        keep_count = max(1, int(round(count * (1.0 - drop_fraction))))
        indices = ctx.rng.choice(count, size=keep_count, replace=False)
        indices.sort()
        return sample.with_lidar_frame(LidarFrame(frame.points[indices].copy(), frame.fields, frame.sensor_model))
