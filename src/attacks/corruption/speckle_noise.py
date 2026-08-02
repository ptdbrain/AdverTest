"""Group A: Speckle noise."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class SpeckleNoise(ImageCorruptionBase):
    """Speckle noise from ImageNet-C."""

    name: ClassVar[str] = "speckle_noise"
