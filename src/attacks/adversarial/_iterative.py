"""Shared projection and gradient helpers for group-D attacks."""

from __future__ import annotations

import numpy as np

from src.adapters.base import ModelAdapter
from src.attacks.base import AttackContext
from src.core.objectives import AttackObjective
from src.core.types import Sample


def input_gradient(
    model: ModelAdapter,
    sample: Sample,
    objective: AttackObjective,
    *,
    allow_zero: bool = False,
) -> np.ndarray:
    gradient = np.asarray(model.input_gradient(sample, objective), dtype=np.float32)
    if gradient.shape != sample.image.shape:
        raise ValueError(
            f"surrogate gradient shape {gradient.shape} != image shape {sample.image.shape}"
        )
    if not np.isfinite(gradient).all():
        raise ValueError("surrogate gradient contains NaN or inf")
    if not allow_zero and not np.any(np.abs(gradient) > 1e-12):
        raise ValueError("surrogate gradient is zero everywhere")
    return gradient


def project_linf(candidate: np.ndarray, original: np.ndarray, epsilon: float) -> np.ndarray:
    delta = np.clip(candidate - original, -epsilon, epsilon)
    return np.clip(original + delta, 0.0, 1.0).astype(np.float32)


def project_l2(candidate: np.ndarray, original: np.ndarray, radius: float) -> np.ndarray:
    delta = candidate.astype(np.float64) - original.astype(np.float64)
    norm = float(np.linalg.norm(delta))
    if norm > radius and norm > 0.0:
        delta *= radius / norm
    return np.clip(original.astype(np.float64) + delta, 0.0, 1.0).astype(np.float32)


def iterative_linf(
    sample: Sample,
    ctx: AttackContext,
    *,
    epsilon: float,
    steps: int,
    step_size: float,
    objective: AttackObjective | None = None,
    random_start: bool = True,
    momentum: float = 0.0,
) -> Sample:
    model = ctx.require_model("iterative_linf")
    original = sample.image
    if random_start:
        perturbation = ctx.rng.uniform(-epsilon, epsilon, size=original.shape).astype(np.float32)
        current = project_linf(original + perturbation, original, epsilon)
    else:
        current = original.copy()
    velocity = np.zeros_like(original, dtype=np.float32)
    selected_objective = objective or ctx.objective
    updates = 0
    for _ in range(max(1, steps)):
        current_sample = sample.with_image(current)
        gradient = input_gradient(
            model,
            current_sample,
            selected_objective,
            allow_zero=True,
        )
        if not np.any(np.abs(gradient) > 1e-12):
            if updates == 0:
                raise ValueError("surrogate gradient is zero everywhere")
            break
        if momentum:
            scale = float(np.mean(np.abs(gradient)))
            normalized = gradient / max(scale, 1e-12)
            velocity = momentum * velocity + normalized
            direction = np.sign(velocity)
        else:
            direction = np.sign(gradient)
        current = project_linf(current + step_size * direction, original, epsilon)
        updates += 1
    return sample.with_image(current)
