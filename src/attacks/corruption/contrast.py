"""Group A: Contrast."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class Contrast(ImageCorruptionBase):
    """Contrast from ImageNet-C."""

    name: ClassVar[str] = "contrast"
