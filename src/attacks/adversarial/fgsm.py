"""Group D: FGSM — the reference white-box plugin.

Copy this file (not ``_template.py``) when your attack needs the model: it shows
the gradient bridge, the L-inf projection, and the severity -> epsilon ladder.

Caveat about the default epsilons: ``{1, 2, 4, 8, 16}/255`` follows plan §2 and
is calibrated for real CNN/transformer detectors, where such a tiny perturbation
is enough. The CI reference model (``blob_detector``) is a threshold detector
whose decision margin is far larger than 16/255, so against *it* the default
ladder barely moves AP. That is a property of the stub, not a broken attack —
verify mechanics against ``blob_detector`` (optionally with a larger epsilon via
``attack_params``), and verify strength against a real adapter.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.adversarial._iterative import input_gradient
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class FgsmParams(AttackParams):
    """Epsilon ladder in 0..1 units; plan §2 uses {1, 2, 4, 8}/255 on 8-bit images."""

    epsilon_per_severity: tuple[float, ...] = (
        1 / 255,
        2 / 255,
        4 / 255,
        8 / 255,
        16 / 255,
    )


@ATTACKS.register
class Fgsm(BaseAttack):
    """One-step L-inf ascent on the adapter's attack loss."""

    name: ClassVar[str] = "fgsm"
    group: ClassVar[AttackGroup] = "D"
    cost_class: ClassVar[CostClass] = "medium"
    needs_model: ClassVar[bool] = True
    needs_gradients: ClassVar[bool] = True
    required_capabilities = frozenset({"input_gradient", "detection_loss"})
    owner: ClassVar[str] = "core"
    reference: ClassVar[str] = "Goodfellow et al., ICLR 2015 (arXiv:1412.6572)"
    params_model: ClassVar[type[AttackParams]] = FgsmParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        epsilon = self.level(severity, self.params.epsilon_per_severity)
        model = ctx.require_model(self.name)
        # loss_for_attack is defined so that *higher* means worse detection,
        # hence a gradient ascent step. The adapter owns the framework.
        gradient = input_gradient(model, sample, ctx.objective)
        step = epsilon * np.sign(gradient)
        return sample.with_image(sample.image + step)
