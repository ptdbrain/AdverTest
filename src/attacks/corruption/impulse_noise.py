"""Group A: Impulse noise."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class ImpulseNoise(ImageCorruptionBase):
    """Impulse noise from ImageNet-C."""

    name: ClassVar[str] = "impulse_noise"
