"""Group D: momentum iterative FGSM for transferable perturbations."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.adversarial._iterative import iterative_linf
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class MiFgsmParams(AttackParams):
    epsilon_per_severity: tuple[float, ...] = (
        1 / 255,
        2 / 255,
        4 / 255,
        8 / 255,
        16 / 255,
    )
    steps: int = 10
    momentum: float = 1.0


@ATTACKS.register
class MiFgsm(BaseAttack):
    """Momentum iterative L-infinity attack."""

    name: ClassVar[str] = "mi_fgsm"
    group: ClassVar[AttackGroup] = "D"
    cost_class: ClassVar[CostClass] = "medium"
    needs_model: ClassVar[bool] = True
    needs_gradients: ClassVar[bool] = True
    required_capabilities = frozenset({"input_gradient", "detection_loss"})
    owner: ClassVar[str] = "group-d-e"
    reference: ClassVar[str] = "Dong et al., CVPR 2018 (arXiv:1710.06081)"
    params_model: ClassVar[type[AttackParams]] = MiFgsmParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        epsilon = self.level(severity, self.params.epsilon_per_severity)
        return iterative_linf(
            sample,
            ctx,
            epsilon=epsilon,
            steps=self.params.steps,
            step_size=epsilon / max(1, self.params.steps),
            random_start=False,
            momentum=self.params.momentum,
        )

