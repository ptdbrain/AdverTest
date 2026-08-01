"""Group D: PGD against a promptable segmentation surrogate."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from src.attacks import ATTACKS
from src.attacks.adversarial._iterative import iterative_linf
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class Sam2PgdParams(AttackParams):
    epsilon_per_severity: tuple[float, ...] = (
        1 / 255,
        2 / 255,
        3 / 255,
        4 / 255,
        8 / 255,
    )
    steps: int = 20


@ATTACKS.register
class Sam2Pgd(BaseAttack):
    """L-infinity PGD maximising segmentation BCE for a box-prompted mask."""

    name: ClassVar[str] = "sam2_pgd"
    group: ClassVar[AttackGroup] = "D"
    cost_class: ClassVar[CostClass] = "medium"
    needs_model: ClassVar[bool] = True
    needs_gradients: ClassVar[bool] = True
    required_annotations = frozenset({"boxes", "mask"})
    required_capabilities = frozenset({"input_gradient", "segmentation_loss"})
    owner: ClassVar[str] = "group-d-e"
    reference: ClassVar[str] = "Custom PGD objective for SAM2 segmentation BCE"
    params_model: ClassVar[type[AttackParams]] = Sam2PgdParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        epsilon = self.level(severity, self.params.epsilon_per_severity)
        return iterative_linf(
            sample,
            ctx,
            epsilon=epsilon,
            steps=self.params.steps,
            step_size=2.5 * epsilon / max(1, self.params.steps),
            objective=replace(ctx.objective, kind="segmentation_bce"),
        )

