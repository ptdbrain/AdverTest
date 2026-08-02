"""Shared logic for ImageNet-C corruptions."""

from __future__ import annotations

import random
from typing import ClassVar
from unittest.mock import patch

import numpy as np

from src.attacks.base import AttackContext, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None


class ImageCorruptionBase(BaseAttack):
    """Base class for all ImageNet-C corruptions."""

    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "core"
    reference: ClassVar[str] = "Hendrycks & Dietterich, ICLR 2019 (imagecorruptions)"

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        if corrupt is None:
            raise RuntimeError("Please install 'imagecorruptions' to use this attack")
        img_uint8 = np.clip(sample.image * 255.0, 0, 255).astype(np.uint8)

        seed = int(ctx.rng.integers(0, 2**31 - 1))
        random.seed(seed)
        np.random.seed(seed)

        orig_rng = np.random.default_rng
        with patch('numpy.random.default_rng', lambda *a, **k: orig_rng(seed)):
            corrupted_uint8 = corrupt(img_uint8, corruption_name=self.name, severity=severity)

        new_image = corrupted_uint8.astype(np.float32) / 255.0
        return sample.with_image(new_image)
