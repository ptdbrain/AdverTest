"""Group A: Gaussian blur."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class GaussianBlur(ImageCorruptionBase):
    """Gaussian blur from ImageNet-C."""

    name: ClassVar[str] = "gaussian_blur"
