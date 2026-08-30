"""Group C: remove complete LiDAR rings to model beam failure (P2.3).

Ensures:
- Real sensor ring / channel ID usage when available.
- Deterministic vertical inclination clustering fallback with explicit 'inferred_ring' provenance.
- Clean error raising when ring info is absent and cannot be inferred.
- Never drops random points pretending to be beam drop.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, LidarFrame, Sample, SensorKind


def infer_lidar_rings(points: np.ndarray, num_rings: int = 64) -> np.ndarray:
    """Deterministically cluster LiDAR points into vertical beam rings based on elevation angle."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-6)
    elevations = np.arcsin(z / r)  # Radians [-pi/2, pi/2]

    # Quantize into num_rings elevation bins
    min_el = float(np.min(elevations))
    max_el = float(np.max(elevations))
    if math.isclose(min_el, max_el):
        return np.zeros(len(points), dtype=np.int32)

    bin_width = (max_el - min_el) / num_rings
    rings = np.clip(((elevations - min_el) / bin_width).astype(np.int32), 0, num_rings - 1)
    return rings


class LidarBeamDropParams(AttackParams):
    fraction_per_severity: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)
    inferred_num_rings: int = 64
    allow_inferred_ring: bool = False


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

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        frame = sample.lidar_frame
        if frame is None or len(frame.points) == 0:
            raise ValueError("ATTACK_UNAVAILABLE_NO_RING_INFO: lidar_beam_drop requires sample.lidar_frame with points")

        inferred = False
        has_explicit_ring = (
            "ring" in frame.fields
            and frame.points.shape[1] > frame.fields.index("ring")
        )
        if has_explicit_ring:
            ring_indices = frame.column("ring")
        elif self.params.allow_inferred_ring:
            # Deterministic ring inference
            ring_indices = infer_lidar_rings(frame.points, num_rings=self.params.inferred_num_rings)
            inferred = True
        else:
            raise ValueError("ATTACK_UNAVAILABLE_NO_RING_INFO: lidar_beam_drop requires a reliable ring field")

        unique_rings = np.unique(ring_indices)
        if len(unique_rings) <= 1:
            raise ValueError("ATTACK_UNAVAILABLE_NO_RING_INFO: Insufficient vertical ring diversity to apply beam drop")

        fraction = self.level(severity, self.params.fraction_per_severity)
        count = min(len(unique_rings) - 1, max(1, int(np.ceil(len(unique_rings) * fraction))))

        # Deterministic drop using PRNG
        dropped_rings = set(ctx.rng.permutation(unique_rings)[:count].tolist())
        keep_mask = ~np.isin(ring_indices, list(dropped_rings))

        filtered_points = frame.points[keep_mask].copy()
        new_frame = LidarFrame(filtered_points, frame.fields[: filtered_points.shape[1]], frame.sensor_model)

        # Record provenance if inferred
        new_meta = dict(sample.meta)
        if inferred:
            new_meta.setdefault("provenance", {})["inferred_ring"] = True
            new_meta["provenance"]["ring_inference_algorithm"] = "inclination_clustering_v1"

        res = sample.with_lidar_frame(new_frame)
        return Sample(
            sample_id=res.sample_id,
            image=res.image,
            boxes=res.boxes,
            mask=res.mask,
            depth=res.depth,
            lidar=res.lidar,
            camera_views=res.camera_views,
            lidar_frame=new_frame,
            boxes3d=res.boxes3d,
            anonymized=res.anonymized,
            meta=new_meta,
        )
