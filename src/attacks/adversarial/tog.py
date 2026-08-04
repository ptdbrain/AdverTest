"""Group D: Targeted Objectness Gradient variants for detectors."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar, Literal

from src.attacks import ATTACKS
from src.attacks.adversarial._iterative import iterative_linf
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class TogParams(AttackParams):
    variant: Literal["vanishing", "fabrication", "mislabeling"] = "vanishing"
    epsilon_per_severity: tuple[float, ...] = (
        1 / 255,
        2 / 255,
        4 / 255,
        8 / 255,
        16 / 255,
    )
    steps: int = 10


@ATTACKS.register
class Tog(BaseAttack):
    """Object vanishing, fabrication, or mislabeling via objectness gradients."""

    name: ClassVar[str] = "tog"
    group: ClassVar[AttackGroup] = "D"
    cost_class: ClassVar[CostClass] = "medium"
    needs_model: ClassVar[bool] = True
    needs_gradients: ClassVar[bool] = True
    required_annotations = frozenset({"boxes"})
    required_capabilities = frozenset(
        {"input_gradient", "objectness", "class_logits"}
    )
    owner: ClassVar[str] = "group-d-e"
    reference: ClassVar[str] = "Chow et al., arXiv:2004.04320"
    params_model: ClassVar[type[AttackParams]] = TogParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        epsilon = self.level(severity, self.params.epsilon_per_severity)
        objective = replace(ctx.objective, kind=self.params.variant)
        return iterative_linf(
            sample,
            ctx,
            epsilon=epsilon,
            steps=self.params.steps,
            step_size=2.5 * epsilon / max(1, self.params.steps),
            objective=objective,
        )
