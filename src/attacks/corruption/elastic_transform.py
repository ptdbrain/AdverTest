"""Group A: Elastic transform."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.corruption._base import ImageCorruptionBase


@ATTACKS.register
class ElasticTransform(ImageCorruptionBase):
    """Elastic transform from ImageNet-C."""

    name: ClassVar[str] = "elastic_transform"
