"""Group C: a frozen sensor keeps serving a frame that is ``k`` keyframes stale.

**Read this before using the numbers.** A real frame freeze replays the *previous*
frame while the world moves on. An attack plugin sees one :class:`Sample` and no
neighbours (``apply`` is a pure function of one sample, per the plugin contract),
so the staleness is *approximated*: the same frame is warped by the ego motion
that would have happened during the freeze — a forward zoom about the image
centre plus a small lateral shift. The result is a frame whose pixels no longer
line up with the ground truth by roughly the right amount, which is what the
metric measures, but it is not literally the previous frame.

The faithful version needs a sequence-aware dataset (KITTI tracking, nuScenes
keyframes) and a plugin contract that can reach neighbouring samples; that is a
separate slot, tracked with the LiDAR/multi-camera group C rows.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.image_ops import nearest_resize
from src.core.types import AttackGroup, CostClass, Sample


class FrameFreezeParams(AttackParams):
    """How many keyframes are missed, and the ego motion assumed per keyframe."""

    stale_frames_per_severity: tuple[int, ...] = (1, 2, 3, 4, 5)
    #: Forward motion: fraction of the frame the scene expands by, per stale frame.
    zoom_per_frame: float = Field(default=0.015, gt=0.0, le=0.5)
    #: Lateral motion: pixels of horizontal drift per stale frame.
    shift_px_per_frame: float = Field(default=2.0, ge=0.0)


@ATTACKS.register
class FrameFreeze(BaseAttack):
    """Stale frame from a frozen sensor, approximated by an ego-motion warp."""

    name: ClassVar[str] = "frame_freeze"
    group: ClassVar[AttackGroup] = "C"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "phong"
    reference: ClassVar[str] = "AdverTest plan §2 group C (frame freeze, single-frame surrogate)"
    params_model: ClassVar[type[AttackParams]] = FrameFreezeParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: FrameFreezeParams = self.params  # type: ignore[assignment]
        # Drawn before severity is read, so every severity of this cell drifts the
        # same way and the ladder stays ordered.
        direction = 1.0 if ctx.rng.random() < 0.5 else -1.0
        stale = self.level(severity, params.stale_frames_per_severity)

        zoomed = self._zoom(sample.image, 1.0 + params.zoom_per_frame * stale)
        drifted = self._translate(zoomed, int(round(params.shift_px_per_frame * stale * direction)))
        return sample.with_image(np.ascontiguousarray(drifted, dtype=np.float32))

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
        """Lateral motion by ``dx`` pixels, replicating the edge column.

        Kept separate from the zoom on purpose: cropping the drift out of the
        zoom margin would silently cap it at a couple of pixels whenever
        ``zoom_per_frame`` is small.
        """
        if dx == 0:
            return image
        width = image.shape[1]
        padded = np.pad(image, ((0, 0), (abs(dx), abs(dx)), (0, 0)), mode="edge")
        start = abs(dx) - dx
        return padded[:, start : start + width]
