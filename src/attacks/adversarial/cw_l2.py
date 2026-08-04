"""Group D: confidence-margin C&W L2 optimisation in tanh space."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np
from pydantic import Field

from src.attacks import ATTACKS
from src.attacks.adversarial._iterative import input_gradient, project_l2
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class CwL2Params(AttackParams):
    radius_per_severity: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    iterations: int = Field(default=100, ge=1)
    binary_search_steps: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
    initial_const: float = Field(default=0.1, gt=0.0)
    confidence: float = Field(default=0.0, ge=0.0)
    adam_beta1: float = Field(default=0.9, ge=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.999, ge=0.0, lt=1.0)


@ATTACKS.register
class CwL2(BaseAttack):
    """L2 + confidence-margin attack with Adam and binary search over c."""

    name: ClassVar[str] = "cw_l2"
    version: ClassVar[str] = "2.0.0"
    group: ClassVar[AttackGroup] = "D"
    cost_class: ClassVar[CostClass] = "expensive"
    needs_model: ClassVar[bool] = True
    needs_gradients: ClassVar[bool] = True
    required_annotations = frozenset({"boxes"})
    required_capabilities = frozenset({"input_gradient", "class_margin"})
    owner: ClassVar[str] = "group-d-e"
    reference: ClassVar[str] = "Carlini & Wagner, IEEE S&P 2017 (arXiv:1608.04644)"
    params_model: ClassVar[type[AttackParams]] = CwL2Params

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        # giải thích từng tham số:
        # radius_per_severity: bán kính tối đa của perturbation theo từng mức độ nghiêm trọng
        # iterations: số lần lặp tối đa để tối ưu hóa perturbation
        # binary_search_steps: số bước tìm kiếm nhị phân để điều chỉnh hằng số
        # learning_rate: tốc độ học cho thuật toán Adam
        # initial_const: hằng số ban đầu để cân bằng giữa loss và perturbation
        model = ctx.require_model(self.name)
        radius = self.level(severity, self.params.radius_per_severity)
        clipped = np.clip(sample.image, 1e-6, 1.0 - 1e-6)
        initial_w = np.arctanh(clipped * 2.0 - 1.0).astype(np.float32)
        margin_objective = replace(ctx.objective, kind="cw_margin")
        best_success: Sample | None = None
        best_success_norm = float("inf")
        best_fallback = sample
        best_fallback_margin = float("-inf")
        best_fallback_norm = float("inf")
        lower_bound = 0.0
        upper_bound = float("inf")
        constant = self.params.initial_const
        for _ in range(max(1, self.params.binary_search_steps)):
            w = initial_w.copy()
            first_moment = np.zeros_like(w)
            second_moment = np.zeros_like(w)
            search_succeeded = False
            for iteration in range(1, max(1, self.params.iterations) + 1):
                tanh_w = np.tanh(w)
                current = ((tanh_w + 1.0) * 0.5).astype(np.float32)
                current = project_l2(current, sample.image, radius)
                candidate = sample.with_image(current)
                margin = model.loss_for_attack(candidate, margin_objective)
                delta = current - sample.image
                l2_norm = float(np.linalg.norm(delta))
                if margin > best_fallback_margin or (
                    np.isclose(margin, best_fallback_margin)
                    and l2_norm < best_fallback_norm
                ):
                    best_fallback = candidate
                    best_fallback_margin = margin
                    best_fallback_norm = l2_norm
                if margin >= self.params.confidence:
                    search_succeeded = True
                    if l2_norm < best_success_norm:
                        best_success = candidate
                        best_success_norm = l2_norm

                margin_gradient = input_gradient(model, candidate, margin_objective)
                pixel_gradient = 2.0 * delta
                if margin < self.params.confidence:
                    pixel_gradient -= constant * margin_gradient
                chain = 2.0 * current * (1.0 - current)
                w_gradient = pixel_gradient * chain
                first_moment = (
                    self.params.adam_beta1 * first_moment
                    + (1.0 - self.params.adam_beta1) * w_gradient
                )
                second_moment = (
                    self.params.adam_beta2 * second_moment
                    + (1.0 - self.params.adam_beta2) * np.square(w_gradient)
                )
                corrected_first = first_moment / (
                    1.0 - self.params.adam_beta1**iteration
                )
                corrected_second = second_moment / (
                    1.0 - self.params.adam_beta2**iteration
                )
                w -= self.params.learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )

            if search_succeeded:
                upper_bound = min(upper_bound, constant)
            else:
                lower_bound = max(lower_bound, constant)
            if np.isfinite(upper_bound):
                constant = (lower_bound + upper_bound) / 2.0
            else:
                constant *= 10.0
        return best_success if best_success is not None else best_fallback
