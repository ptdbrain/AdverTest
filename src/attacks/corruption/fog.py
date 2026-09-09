"""Group A: Fog."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class Fog(ImageCorruptionBase):
    """Fog from ImageNet-C."""

    name: ClassVar[str] = "fog"
