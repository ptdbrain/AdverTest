"""Per-attack test: the specifics the shared contract test cannot know.

Pattern to copy for your own attack — one file per attack, named after it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.types import Sample


def _flat_sample() -> Sample:
    """Uniform grey frame, so the measured noise is only what the attack added."""
    return Sample("flat", np.full((64, 64, 3), 0.5, dtype=np.float32))


@pytest.mark.parametrize("severity", [1, 3, 5])
def test_noise_std_matches_the_configured_sigma(severity: int) -> None:
    attack = get_attack("gaussian_noise")
    context = AttackContext(rng=np.random.default_rng(0))
    sample = _flat_sample()
    attacked = attack.run(sample, severity, context)
    expected = attack.params.sigma_per_severity[severity - 1]
    assert float(np.std(attacked.image - sample.image)) == pytest.approx(expected, rel=0.1)


def test_sigma_is_configurable_per_run() -> None:
    attack = get_attack("gaussian_noise", sigma_per_severity=(0.5,))
    attacked = attack.run(_flat_sample(), 1, AttackContext(rng=np.random.default_rng(0)))
    assert float(np.std(attacked.image - _flat_sample().image)) > 0.1


def test_values_stay_inside_the_valid_range() -> None:
    """Clipping is done by ``BaseAttack.run``, even with an absurd sigma."""
    attacked = get_attack("gaussian_noise", sigma_per_severity=(5.0,)).run(
        _flat_sample(), 1, AttackContext(rng=np.random.default_rng(0))
    )
    assert attacked.image.min() >= 0.0 and attacked.image.max() <= 1.0
