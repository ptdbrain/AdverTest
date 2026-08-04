"""Group C: deterministic dropout of one or more nuScenes camera views."""

from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, CameraView, SensorKind


class CameraDropoutParams(AttackParams):
    cameras_dropped_per_severity: tuple[int, ...] = (1, 2, 3, 4, 5)
    mode: Literal["black", "freeze"] = "black"
    camera_names: tuple[str, ...] = (
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    )
    black_value: float = Field(default=0.0, ge=0.0, le=1.0)


@ATTACKS.register
class CameraDropout(BaseAttack):
    """Drop a deterministic subset of the six-camera rig."""

    name: ClassVar[str] = "camera_dropout"
    group: ClassVar[AttackGroup] = "C"
    modality: ClassVar[str] = "multi"
    cost_class: ClassVar[CostClass] = "cheap"
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"camera_rig"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"camera_rig", "image"})
    params_model: ClassVar[type[AttackParams]] = CameraDropoutParams

    def apply(self, sample, severity, ctx):
        params: CameraDropoutParams = self.params  # type: ignore[assignment]
        views = list(sample.camera_views)
        candidates = [i for i, view in enumerate(views) if view.name in params.camera_names]
        count = min(len(candidates) - 1, int(self.level(severity, params.cameras_dropped_per_severity)))
        dropped = set(ctx.rng.permutation(candidates)[:max(0, count)].tolist())
        updated = []
        for index, view in enumerate(views):
            if index not in dropped:
                updated.append(view)
                continue
            if params.mode == "freeze":
                if view.previous_image is None:
                    raise ValueError("camera_dropout(mode='freeze') requires previous_image")
                image = view.previous_image
            else:
                image = np.full_like(view.image, params.black_value, dtype=np.float32)
            updated.append(CameraView(view.name, image, view.depth, view.intrinsic, view.sensor_to_ego, view.previous_image))
        return sample.with_camera_views(updated)
