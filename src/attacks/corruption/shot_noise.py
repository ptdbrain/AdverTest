"""Group A: Shot noise."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class ShotNoise(ImageCorruptionBase):
    """Shot noise from ImageNet-C."""

    name: ClassVar[str] = "shot_noise"
