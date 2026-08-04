"""Group F: Square Attack — query-efficient score-based black-box attack.

A derivative-free adversarial attack that iteratively perturbs random square
regions of the image.  At each step the attack evaluates whether the new
perturbation degrades detection more than the previous best; if so it keeps
the change, otherwise it reverts.  No gradient is ever requested, which makes
this attack work against adapters with ``supports_gradients = False`` and
catches gradient-masking defences that deceive Group D attacks (plan §11).

The square size starts large (controlled by ``p_init``) and shrinks over the
query budget so early iterations explore globally while later ones refine.

Reference parameters from the plan:
    ε = 8/255,  query budget ∈ {500, 1000, 2500}

Severity controls *both* ε and query budget so the contract's monotonicity
requirement (stronger severity ⇒ larger perturbation norm) is satisfied
even when the stochastic search gets unlucky.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.adapters.base import ModelAdapter
from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class SquareAttackParams(AttackParams):
    """Tunable knobs — one value per severity level."""

    epsilon_per_severity: tuple[float, ...] = (
        2 / 255,
        4 / 255,
        8 / 255,
        16 / 255,
        32 / 255,
    )
    queries_per_severity: tuple[int, ...] = (100, 250, 500, 1000, 2500)
    #: Fraction of the image area covered by the first square.
    p_init: float = 0.8


@ATTACKS.register
class SquareAttack(BaseAttack):
    """Score-based black-box L-inf attack via random square perturbations."""

    name: ClassVar[str] = "square_attack"
    group: ClassVar[AttackGroup] = "F"
    cost_class: ClassVar[CostClass] = "expensive"
    needs_model: ClassVar[bool] = True
    needs_gradients: ClassVar[bool] = False
    owner: ClassVar[str] = "nguyenhuucong"
    reference: ClassVar[str] = "Andriushchenko et al., ECCV 2020 (arXiv:1912.00049)"
    params_model: ClassVar[type[AttackParams]] = SquareAttackParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        epsilon = self.level(severity, self.params.epsilon_per_severity)
        n_queries = int(self.level(severity, self.params.queries_per_severity))
        model = ctx.require_model(self.name)

        h, w, c = sample.image.shape
        x_orig = sample.image

        # ---- initialise with a random perturbation inside the ε-ball ----
        delta = ctx.rng.uniform(-epsilon, epsilon, (h, w, c)).astype(np.float32)
        x_best = np.clip(x_orig + delta, 0.0, 1.0).astype(np.float32)
        best_score = self._detection_score(model, sample, x_best)

        # ---- iterative square perturbation ----
        for i in range(n_queries):
            # Shrink the square over iterations (paper §3.1).
            p = self.params.p_init * (1.0 - float(i) / max(n_queries, 1))
            side = max(1, int(round(np.sqrt(p * h * w))))
            side = min(side, h, w)

            y0 = int(ctx.rng.integers(0, max(1, h - side + 1)))
            x0 = int(ctx.rng.integers(0, max(1, w - side + 1)))

            # Candidate: replace the square region with fresh random noise
            # while staying within the ε-ball of the *original* image.
            x_cand = x_best.copy()
            new_delta = ctx.rng.uniform(
                -epsilon, epsilon, (side, side, c),
            ).astype(np.float32)
            x_cand[y0 : y0 + side, x0 : x0 + side] = np.clip(
                x_orig[y0 : y0 + side, x0 : x0 + side] + new_delta,
                0.0,
                1.0,
            ).astype(np.float32)

            cand_score = self._detection_score(model, sample, x_cand)
            if cand_score <= best_score:
                x_best = x_cand
                best_score = cand_score

        return sample.with_image(x_best)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _detection_score(
        model: ModelAdapter,
        sample: Sample,
        image: np.ndarray,
    ) -> float:
        """Sum of detection confidence scores — lower means a more successful attack.

        Using the raw confidence sum gives a smooth signal for the search
        (compared to e.g. "number of detections" which is very coarse).
        """
        predictions = model.predict([sample.with_image(image)])
        if not predictions or not predictions[0].boxes:
            return 0.0
        return sum(box.score for box in predictions[0].boxes)
