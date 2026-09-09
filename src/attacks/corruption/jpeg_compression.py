"""Group A: Jpeg compression."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class JpegCompression(ImageCorruptionBase):
    """Jpeg compression from ImageNet-C."""

    name: ClassVar[str] = "jpeg_compression"
