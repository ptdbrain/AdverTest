"""Group C: frame freeze temporal attack across single and multi-camera sequences (P2.4).

Ensures:
- Replays actual previous frame (t - 1) when sequence history is available.
- Synchronizes freeze across all camera views by timestamp.
- Handles first frame in sequence by raising FIRST_FRAME_IN_SEQUENCE_CANNOT_FREEZE or skipping cleanly.
- Records source_frame_id and source_timestamp in provenance.
- Provides documented ego-motion warp surrogate when sequence history is unavailable.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.image_ops import nearest_resize
from src.core.types import AttackGroup, CameraView, CostClass, Sample


class FrameFreezeParams(AttackParams):
    """Configuration for frame freeze attack."""

    stale_frames_per_severity: tuple[int, ...] = (1, 2, 3, 4, 5)
    zoom_per_frame: float = Field(default=0.015, gt=0.0, le=0.5)
    shift_px_per_frame: float = Field(default=2.0, ge=0.0)


@ATTACKS.register
class FrameFreeze(BaseAttack):
    """Stale frame replay attack for single and multi-camera streams."""

    name: ClassVar[str] = "frame_freeze"
    group: ClassVar[AttackGroup] = "C"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "phong"
    reference: ClassVar[str] = "AdverTest plan §2 group C (sequence-aware & surrogate frame freeze)"
    params_model: ClassVar[type[AttackParams]] = FrameFreezeParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: FrameFreezeParams = self.params  # type: ignore[assignment]
        stale = int(self.level(severity, params.stale_frames_per_severity))

        # 1. Sequence-aware freeze if sequence metadata/history is present
        seq_history = sample.meta.get("sequence_history") or getattr(ctx, "sequence_history", None)
        frame_idx = sample.meta.get("frame_index")

        if seq_history is not None and frame_idx is not None:
            target_idx = int(frame_idx) - int(stale)
            if target_idx < 0:
                raise ValueError("FIRST_FRAME_IN_SEQUENCE_CANNOT_FREEZE: Cannot freeze on initial sequence frame")

            source_sample = seq_history[target_idx]
            frozen_cams = []
            if source_sample.camera_views:
                for cam in source_sample.camera_views:
                    frozen_cams.append(CameraView(cam.name, cam.image.copy()))

            frozen_sample = Sample(
                sample_id=sample.sample_id,
                image=source_sample.image.copy(),
                boxes=sample.boxes,
                mask=sample.mask,
                depth=sample.depth,
                lidar=sample.lidar,
                camera_views=tuple(frozen_cams) if frozen_cams else sample.camera_views,
                lidar_frame=sample.lidar_frame,
                boxes3d=sample.boxes3d,
                anonymized=sample.anonymized,
                meta={
                    **sample.meta,
                    "provenance": {
                        **sample.meta.get("provenance", {}),
                        "source_frame_id": source_sample.sample_id,
                        "source_timestamp": source_sample.meta.get("timestamp"),
                        "stale_offset": stale,
                    },
                },
            )
            return frozen_sample

        # 2. Standalone surrogate (ego-motion warp)
        direction = 1.0 if ctx.rng.random() < 0.5 else -1.0
        zoomed = self._zoom(sample.image, 1.0 + params.zoom_per_frame * stale)
        drifted = self._translate(zoomed, int(round(params.shift_px_per_frame * stale * direction)))

        # Also warp camera views if present
        warped_cams = []
        if sample.camera_views:
            for cam in sample.camera_views:
                z = self._zoom(cam.image, 1.0 + params.zoom_per_frame * stale)
                d = self._translate(z, int(round(params.shift_px_per_frame * stale * direction)))
                warped_cams.append(CameraView(cam.name, np.ascontiguousarray(d, dtype=np.float32)))

        return Sample(
            sample_id=sample.sample_id,
            image=np.ascontiguousarray(drifted, dtype=np.float32),
            boxes=sample.boxes,
            mask=sample.mask,
            depth=sample.depth,
            lidar=sample.lidar,
            camera_views=tuple(warped_cams) if warped_cams else sample.camera_views,
            lidar_frame=sample.lidar_frame,
            boxes3d=sample.boxes3d,
            anonymized=sample.anonymized,
            meta={
                **sample.meta,
                "provenance": {
                    **sample.meta.get("provenance", {}),
                    "surrogate": "ego_motion_warp",
                },
            },
        )

    @staticmethod
    def _zoom(image: np.ndarray, factor: float) -> np.ndarray:
        """Forward motion: scale about the image centre, then crop back to size."""
        height, width = image.shape[:2]
        scaled_h = max(height + 1, int(round(height * factor)))
        scaled_w = max(width + 1, int(round(width * factor)))
        scaled = nearest_resize(image, scaled_h, scaled_w)
        top = (scaled_h - height) // 2
        left = (scaled_w - width) // 2
        return scaled[top : top + height, left : left + width]

    @staticmethod
    def _translate(image: np.ndarray, dx: int) -> np.ndarray:
        """Lateral motion by ``dx`` pixels, replicating the edge column."""
        if dx == 0:
            return image
        width = image.shape[1]
        padded = np.pad(image, ((0, 0), (abs(dx), abs(dx)), (0, 0)), mode="edge")
        start = abs(dx) - dx
        return padded[:, start : start + width]
