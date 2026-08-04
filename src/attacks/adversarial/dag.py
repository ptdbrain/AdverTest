"""Group D: dense adversary generation for proposal-based detectors."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.adversarial._iterative import iterative_linf
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class DagParams(AttackParams):
    epsilon_per_severity: tuple[float, ...] = (
        1 / 255,
        2 / 255,
        4 / 255,
        8 / 255,
        16 / 255,
    )
    iterations: int = 150


@ATTACKS.register
class Dag(BaseAttack):
    """Iterative dense-proposal attack for Faster R-CNN-style surrogates."""

    name: ClassVar[str] = "dag"
    group: ClassVar[AttackGroup] = "D"
    cost_class: ClassVar[CostClass] = "expensive"
    needs_model: ClassVar[bool] = True
    needs_gradients: ClassVar[bool] = True
    required_annotations = frozenset({"boxes"})
    required_capabilities = frozenset({"input_gradient", "dense_proposals"})
    owner: ClassVar[str] = "group-d-e"
    reference: ClassVar[str] = "Xie et al., ICCV 2017 (arXiv:1703.08603)"
    params_model: ClassVar[type[AttackParams]] = DagParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        epsilon = self.level(severity, self.params.epsilon_per_severity)
        return iterative_linf(
            sample,
            ctx,
            epsilon=epsilon,
            steps=self.params.iterations,
            step_size=epsilon / max(1, self.params.iterations),
            objective=replace(ctx.objective, kind="dag"),
        )

