from __future__ import annotations

import numpy as np
import pytest

from src.attacks import get_attack
from src.attacks.recipes import AttackRecipe, AttackRecipeStep
from src.core.hashing import array_digest
from src.core.types import Box, Sample
from src.pipeline.composition import (
    CompositionContext,
    CompositionEngine,
    CompositionExecutionError,
)


def _sample() -> Sample:
    image = np.linspace(0.0, 1.0, 40 * 48 * 3, dtype=np.float32).reshape(
        40,
        48,
        3,
    )
    return Sample(
        sample_id="composition-1",
        image=image,
        boxes=(Box(4, 6, 30, 34, "Car"),),
        anonymized=True,
    )


def _step(position: int, name: str, seed: int, *, probability: float = 1.0, **params):
    return AttackRecipeStep(
        position=position,
        attack_name=name,
        implementation_version="1.0.0",
        severity=3,
        parameters=params,
        probability=probability,
        seed=seed,
        expected_cost=1.0,
    )


def test_order_changes_output_and_same_recipe_is_byte_deterministic() -> None:
    fog_then_jpeg = AttackRecipe(
        name="fog-then-jpeg",
        steps=(_step(0, "fog", 11), _step(1, "jpeg_compression", 12)),
    )
    jpeg_then_fog = AttackRecipe(
        name="jpeg-then-fog",
        steps=(_step(0, "jpeg_compression", 11), _step(1, "fog", 12)),
    )
    engine = CompositionEngine()
    context = CompositionContext(run_seed=195)

    first = engine.execute(_sample(), fog_then_jpeg, context)
    repeated = engine.execute(_sample(), fog_then_jpeg, context)
    reversed_order = engine.execute(_sample(), jpeg_then_fog, context)

    assert first.loadable and first.final_sample is not None
    assert repeated.final_sample is not None
    assert array_digest(first.final_sample.image) == array_digest(
        repeated.final_sample.image
    )
    assert first.step_records == repeated.step_records
    assert first.intermediate_hashes == repeated.intermediate_hashes
    assert array_digest(first.final_sample.image) != array_digest(
        reversed_order.final_sample.image  # type: ignore[union-attr]
    )


def test_attack_exposes_resolved_severity_parameters() -> None:
    attack = get_attack("gaussian_noise")

    resolved = attack.resolve_parameters(2)

    assert resolved["severity"] == 2
    assert resolved["sigma"] == attack.level(
        2,
        attack.params.sigma_per_severity,
    )


def test_probability_zero_is_recorded_not_silently_skipped() -> None:
    recipe = AttackRecipe(
        name="probability-zero",
        steps=(_step(0, "gaussian_noise", 1, probability=0.0),),
    )

    result = CompositionEngine().execute(
        _sample(),
        recipe,
        CompositionContext(run_seed=1),
    )

    assert result.loadable
    assert result.step_records[0].status == "not_selected_by_probability"
    assert result.step_records[0].applied is False
    assert result.intermediate_hashes[0] == array_digest(_sample().image)


def test_non_fail_fast_records_failure_and_blocks_remaining_steps() -> None:
    recipe = AttackRecipe(
        name="invalid-params",
        steps=(
            _step(0, "gaussian_noise", 1, definitely_invalid=True),
            _step(1, "contrast", 2),
        ),
    )

    result = CompositionEngine().execute(
        _sample(),
        recipe,
        CompositionContext(run_seed=1, fail_fast=False),
    )

    assert result.loadable is False
    assert result.final_sample is None
    assert result.step_records[0].status == "failed"
    assert result.step_records[1].status == "blocked_by_prior_failure"
    assert result.errors


def test_fail_fast_raises_with_non_loadable_partial_result() -> None:
    recipe = AttackRecipe(
        name="over-budget",
        steps=(_step(0, "gaussian_noise", 1),),
    )

    with pytest.raises(CompositionExecutionError) as captured:
        CompositionEngine().execute(
            _sample(),
            recipe,
            CompositionContext(run_seed=1, max_cost=0.5, fail_fast=True),
        )

    assert captured.value.result.final_sample is None
    assert captured.value.result.loadable is False
    assert captured.value.result.step_records[0].status == "failed"
