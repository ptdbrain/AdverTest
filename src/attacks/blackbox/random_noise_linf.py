"""Group F: Random L-inf noise baseline.

Uniform random perturbation within an L∞ ball.  This is the simplest possible
"attack" and serves as a **trustworthiness baseline** for Group D (plan §11):
if a gradient-based attack such as PGD is not clearly stronger than random
noise at the *same* ε, then the adversarial attack has not converged or
gradient masking is hiding real vulnerability.

Also useful as a cheap sanity probe: any model whose AP does not drop at all
under random noise at ε = 16/255 is suspicious (thresholding? rounding?).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class RandomNoiseLinfParams(AttackParams):
    """Epsilon ladder matching the Group D default so comparisons are direct."""

    epsilon_per_severity: tuple[float, ...] = (
        1 / 255,
        2 / 255,
        4 / 255,
        8 / 255,
        16 / 255,
    )


@ATTACKS.register
class RandomNoiseLinf(BaseAttack):
    """Uniform random noise within an L-inf ball — Group D trustworthiness baseline."""

    name: ClassVar[str] = "random_noise_linf"
    group: ClassVar[AttackGroup] = "F"
    cost_class: ClassVar[CostClass] = "cheap"
    needs_model: ClassVar[bool] = False
    needs_gradients: ClassVar[bool] = False
    owner: ClassVar[str] = "nguyenhuucong"
    reference: ClassVar[str] = "Baseline: plan §11 — random noise at same ε as adversarial attacks"
    params_model: ClassVar[type[AttackParams]] = RandomNoiseLinfParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        epsilon = self.level(severity, self.params.epsilon_per_severity)
        noise = ctx.rng.uniform(-epsilon, epsilon, size=sample.image.shape).astype(
            np.float32,
        )
        return sample.with_image(sample.image + noise)
