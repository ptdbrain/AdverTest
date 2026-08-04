"""Per-attack tests for the Square Attack black-box plugin."""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.base import ModelAdapter
from src.attacks import get_attack
from src.attacks.base import AttackContext, ModelRequiredError
from src.core.types import Sample


def test_perturbation_stays_within_epsilon_ball(
    adapter: ModelAdapter, sample: Sample,
) -> None:
    """Every pixel must lie within [-ε, ε] of the original."""
    # Use small query budget for speed in tests.
    attack = get_attack("square_attack", queries_per_severity=(5, 5, 5, 5, 5))
    for severity, epsilon in enumerate(attack.params.epsilon_per_severity, start=1):
        ctx = AttackContext(rng=np.random.default_rng(42), model=adapter)
        attacked = attack.run(sample, severity, ctx)
        linf = float(np.max(np.abs(attacked.image - sample.image)))
        assert linf <= epsilon + 1e-6, (
            f"severity {severity}: L-inf {linf:.6f} exceeds ε = {epsilon:.6f}"
        )


def test_running_without_a_model_fails_clearly(sample: Sample) -> None:
    """Square Attack is query-based and must have a model in the context."""
    with pytest.raises(ModelRequiredError, match="needs a model adapter"):
        attack = get_attack("square_attack", queries_per_severity=(2, 2, 2, 2, 2))
        attack.run(sample, 1, AttackContext(rng=np.random.default_rng(0)))


def test_detection_score_decreases_or_stays(
    adapter: ModelAdapter, sample: Sample,
) -> None:
    """After optimisation the detection score should be <= the initial score."""
    attack = get_attack("square_attack", queries_per_severity=(20, 20, 20, 20, 20))
    ctx = AttackContext(rng=np.random.default_rng(0), model=adapter)
    attacked = attack.run(sample, 5, ctx)

    clean_preds = adapter.predict([sample])
    atk_preds = adapter.predict([attacked])

    clean_conf = sum(b.score for b in clean_preds[0].boxes) if clean_preds[0].boxes else 0
    atk_conf = sum(b.score for b in atk_preds[0].boxes) if atk_preds[0].boxes else 0
    assert atk_conf <= clean_conf + 1e-6, (
        "attack should not increase total detection confidence"
    )


def test_reproducible_with_same_seed(
    adapter: ModelAdapter, sample: Sample,
) -> None:
    """Same seed + same queries must produce identical pixels."""
    attack = get_attack("square_attack", queries_per_severity=(10, 10, 10, 10, 10))
    img1 = attack.run(
        sample, 3, AttackContext(rng=np.random.default_rng(99), model=adapter),
    ).image
    img2 = attack.run(
        sample, 3, AttackContext(rng=np.random.default_rng(99), model=adapter),
    ).image
    np.testing.assert_array_equal(img1, img2)
