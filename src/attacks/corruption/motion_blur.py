"""Group A: Motion blur."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class MotionBlur(ImageCorruptionBase):
    """Motion blur from ImageNet-C."""

    name: ClassVar[str] = "motion_blur"
