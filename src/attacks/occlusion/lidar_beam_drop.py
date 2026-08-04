"""Group C: remove complete LiDAR rings to model beam failure."""

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, SensorKind


class LidarBeamDropParams(AttackParams):
    fraction_per_severity: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


@ATTACKS.register
class LidarBeamDrop(BaseAttack):
    """Remove complete sensor rings while preserving point fields."""

    name: ClassVar[str] = "lidar_beam_drop"
    group: ClassVar[AttackGroup] = "C"
    modality: ClassVar[str] = "lidar"
    cost_class: ClassVar[CostClass] = "cheap"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"lidar"})
    owner: ClassVar[str] = "group-c"
    params_model: ClassVar[type[AttackParams]] = LidarBeamDropParams

    def apply(self, sample, severity, ctx):
        frame = sample.lidar_frame
        if frame is None or "ring" not in frame.fields:
            raise ValueError("lidar_beam_drop requires a ring field")
        rings = np.unique(frame.column("ring"))
        count = min(len(rings) - 1, int(np.ceil(len(rings) * self.level(severity, self.params.fraction_per_severity))))
        dropped = set(ctx.rng.permutation(rings)[:max(0, count)].tolist())
        keep = ~np.isin(frame.column("ring"), list(dropped))
        return sample.with_lidar_frame(type(frame)(frame.points[keep].copy(), frame.fields, frame.sensor_model))
