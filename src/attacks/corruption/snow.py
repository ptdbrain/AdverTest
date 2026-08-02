"""Group A: Snow."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class Snow(ImageCorruptionBase):
    """Snow from ImageNet-C."""

    name: ClassVar[str] = "snow"
