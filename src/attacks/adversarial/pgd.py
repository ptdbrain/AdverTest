"""Group D: projected-gradient descent under an L-infinity budget."""

from __future__ import annotations

from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.adversarial._iterative import iterative_linf
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class PgdParams(AttackParams):
    epsilon_per_severity: tuple[float, ...] = (
        1 / 255,
        2 / 255,
        4 / 255,
        8 / 255,
        16 / 255,
    )
    steps: int = 20
    step_ratio: float = 2.5
    random_start: bool = True
    restarts: int = 1


@ATTACKS.register
class Pgd(BaseAttack):
    """Iterative L-infinity attack with random start and projection."""

    name: ClassVar[str] = "pgd"
    group: ClassVar[AttackGroup] = "D"
    cost_class: ClassVar[CostClass] = "medium"
    needs_model: ClassVar[bool] = True
    needs_gradients: ClassVar[bool] = True
    required_capabilities = frozenset({"input_gradient", "detection_loss"})
    owner: ClassVar[str] = "group-d-e"
    reference: ClassVar[str] = "Madry et al., ICLR 2018 (arXiv:1706.06083)"
    params_model: ClassVar[type[AttackParams]] = PgdParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        epsilon = self.level(severity, self.params.epsilon_per_severity)
        steps = max(1, self.params.steps)
        step_size = self.params.step_ratio * epsilon / steps
        best = sample
        best_loss = float("-inf")
        model = ctx.require_model(self.name)
        for _ in range(max(1, self.params.restarts)):
            candidate = iterative_linf(
                sample,
                ctx,
                epsilon=epsilon,
                steps=steps,
                step_size=step_size,
                random_start=self.params.random_start,
            )
            loss = model.loss_for_attack(candidate, ctx.objective)
            if loss > best_loss:
                best, best_loss = candidate, loss
        return best

