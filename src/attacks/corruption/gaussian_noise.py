"""Group A: Additive Gaussian sensor noise."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class GaussianNoiseParams(AttackParams):
    sigma_per_severity: tuple[float, ...] = (0.04, 0.06, 0.09, 0.13, 0.18)


@ATTACKS.register
class GaussianNoise(BaseAttack):
    """Additive Gaussian noise, ImageNet-C severity ladder."""

    name: ClassVar[str] = "gaussian_noise"
    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "core"
    reference: ClassVar[str] = "Hendrycks & Dietterich, ICLR 2019 (arXiv:1903.12261)"
    params_model: ClassVar[type[AttackParams]] = GaussianNoiseParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        sigma = self.level(severity, self.params.sigma_per_severity)
        noise = ctx.rng.normal(0, sigma, sample.image.shape)
        return sample.with_image(sample.image + noise.astype(np.float32))
