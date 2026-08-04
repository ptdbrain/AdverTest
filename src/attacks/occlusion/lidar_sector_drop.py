"""Group C: remove a contiguous azimuth sector from a LiDAR scan."""

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, SensorKind


class LidarSectorDropParams(AttackParams):
    degrees_per_severity: tuple[float, ...] = (30.0, 60.0, 90.0, 135.0, 180.0)


@ATTACKS.register
class LidarSectorDrop(BaseAttack):
    """Remove one deterministic contiguous azimuth sector."""

    name: ClassVar[str] = "lidar_sector_drop"
    group: ClassVar[AttackGroup] = "C"
    modality: ClassVar[str] = "lidar"
    cost_class: ClassVar[CostClass] = "cheap"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    owner: ClassVar[str] = "group-c"
    params_model: ClassVar[type[AttackParams]] = LidarSectorDropParams

    def apply(self, sample, severity, ctx):
        frame = sample.lidar_frame
        if frame is None:
            raise ValueError("lidar_sector_drop requires sample.lidar_frame")
        radians = np.deg2rad(self.level(severity, self.params.degrees_per_severity))
        azimuth = np.mod(np.arctan2(frame.points[:, 1], frame.points[:, 0]), 2.0 * np.pi)
        # Anchor the sector on a real return so sparse scans cannot
        # accidentally produce a no-op at low severity.
        center = float(azimuth[0]) if azimuth.size else 0.0
        distance = np.abs(np.angle(np.exp(1j * (azimuth - center))))
        keep = distance > radians / 2.0
        if not np.any(keep) and keep.size:
            keep[int(np.argmax(distance))] = True
        return sample.with_lidar_frame(type(frame)(frame.points[keep].copy(), frame.fields, frame.sensor_model))
