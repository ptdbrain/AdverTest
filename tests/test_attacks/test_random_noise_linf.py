"""Per-attack tests for the random_noise_linf baseline."""

from __future__ import annotations

import numpy as np
import pytest

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.types import Sample


def test_perturbation_stays_within_epsilon_ball(sample: Sample) -> None:
    """Every pixel must lie within [-ε, ε] of the original."""
    attack = get_attack("random_noise_linf")
    for severity, epsilon in enumerate(attack.params.epsilon_per_severity, start=1):
        ctx = AttackContext(rng=np.random.default_rng(42))
        attacked = attack.run(sample, severity, ctx)
        linf = float(np.max(np.abs(attacked.image - sample.image)))
        assert linf <= epsilon + 1e-6, (
            f"severity {severity}: L-inf {linf:.6f} exceeds ε = {epsilon:.6f}"
        )


def test_mean_perturbation_is_roughly_zero(sample: Sample) -> None:
    """Uniform[-ε, ε] has mean 0; a large bias would indicate a bug."""
    attack = get_attack("random_noise_linf")
    ctx = AttackContext(rng=np.random.default_rng(0))
    attacked = attack.run(sample, 5, ctx)
    diff = attacked.image - sample.image
    assert abs(float(diff.mean())) < 0.02, "mean perturbation should be near zero"


def test_perturbation_norm_grows_with_severity(sample: Sample) -> None:
    """Higher severity (larger ε) must produce a larger average perturbation."""
    attack = get_attack("random_noise_linf")
    norms = []
    for severity in range(1, 6):
        ctx = AttackContext(rng=np.random.default_rng(0))
        attacked = attack.run(sample, severity, ctx)
        norms.append(float(np.linalg.norm(attacked.image - sample.image)))
    for i in range(len(norms) - 1):
        assert norms[i + 1] > norms[i], (
            f"norm at severity {i + 2} ({norms[i + 1]:.4f}) should exceed "
            f"severity {i + 1} ({norms[i]:.4f})"
        )


def test_no_model_required(sample: Sample) -> None:
    """The random noise baseline must work without a model in the context."""
    attack = get_attack("random_noise_linf")
    ctx = AttackContext(rng=np.random.default_rng(0), model=None)
    attacked = attack.run(sample, 3, ctx)
    assert attacked.image.shape == sample.image.shape
