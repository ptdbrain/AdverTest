"""Group A: Zoom blur."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class ZoomBlur(ImageCorruptionBase):
    """Zoom blur from ImageNet-C."""

    name: ClassVar[str] = "zoom_blur"
