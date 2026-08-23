"""Group A: Gaussian coordinate noise on LiDAR XYZ channels."""
from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.base import AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, LidarFrame, SensorKind


class LidarXYZNoiseParams(AttackParams):
    sigma_per_severity: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20)

@ATTACKS.register
class LidarXYZNoise(BaseAttack):
    name: ClassVar[str] = "lidar_xyz_noise"
    group: ClassVar[AttackGroup] = "A"
    modality: ClassVar[str] = "lidar"
    cost_class: ClassVar[CostClass] = "cheap"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    owner: ClassVar[str] = "3d-evaluation"
    params_model: ClassVar[type[AttackParams]] = LidarXYZNoiseParams

    def apply(self, sample, severity, ctx):
        frame = sample.lidar_frame
        if frame is None:
            raise ValueError("lidar_xyz_noise requires sample.lidar_frame")
        sigma = self.level(severity, self.params.sigma_per_severity)
        points = frame.points.copy()
        noise = ctx.rng.normal(loc=0.0, scale=sigma, size=points[:, :3].shape)
        points[:, :3] += noise.astype(points.dtype)
        return sample.with_lidar_frame(
            LidarFrame(points, frame.fields, frame.sensor_model)
        )
