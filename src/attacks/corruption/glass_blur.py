"""Group A: Glass blur."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class GlassBlur(ImageCorruptionBase):
    """Glass blur from ImageNet-C."""

    name: ClassVar[str] = "glass_blur"
