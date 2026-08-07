from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.recipes import (
    AttackRecipe,
    AttackRecipeStep,
    RandomNRequest,
    RecipeBuilder,
    RecipeConstraints,
    StratifiedRandomRequest,
    SweepRequest,
)


def _step(
    position: int,
    name: str = "gaussian_noise",
    *,
    version: str = "1.0.0",
    severity: int = 2,
    seed: int = 7,
    parameters: dict | None = None,
    expected_cost: float = 1.0,
) -> AttackRecipeStep:
    return AttackRecipeStep(
        position=position,
        attack_name=name,
        implementation_version=version,
        severity=severity,
        parameters=parameters or {},
        seed=seed,
        expected_cost=expected_cost,
    )


def test_recipe_hash_is_canonical_and_sensitive_to_execution_fields() -> None:
    first = AttackRecipe(
        name="fixture",
        catalog_version="1.0.0",
        steps=(
            _step(0, parameters={"sigma": 0.1, "nested": {"b": 2, "a": 1}}),
            _step(1, "contrast", seed=8),
        ),
    )
    reordered_parameters = AttackRecipe(
        name="fixture",
        catalog_version="1.0.0",
        steps=(
            _step(0, parameters={"nested": {"a": 1, "b": 2}, "sigma": 0.1}),
            _step(1, "contrast", seed=8),
        ),
    )

    assert first.recipe_hash == reordered_parameters.recipe_hash
    assert first.recipe_id == reordered_parameters.recipe_id

    variants = (
        AttackRecipe(
            name="fixture",
            catalog_version="2.0.0",
            steps=first.steps,
        ),
        AttackRecipe(
            name="fixture",
            steps=(
                _step(0, "contrast", seed=8),
                _step(1, parameters={"sigma": 0.1, "nested": {"b": 2, "a": 1}}),
            ),
        ),
        AttackRecipe(name="fixture", steps=(_step(0, seed=99), _step(1, "contrast", seed=8))),
        AttackRecipe(name="fixture", steps=(_step(0, version="9.0.0"), _step(1, "contrast", seed=8))),
        AttackRecipe(name="fixture", steps=(_step(0, severity=3), _step(1, "contrast", seed=8))),
    )
    assert all(variant.recipe_hash != first.recipe_hash for variant in variants)


def test_recipe_positions_must_be_contiguous_and_unique() -> None:
    with pytest.raises(ValidationError, match="positions"):
        AttackRecipe(name="bad", steps=(_step(1),))


def test_recipe_validation_reports_hard_failures_and_caps() -> None:
    load_attacks()
    recipe = AttackRecipe(
        name="invalid",
        steps=(
            _step(0, "fgsm", expected_cost=4),
            _step(1, "pgd", expected_cost=4),
            _step(
                2,
                "object_occlusion",
                parameters={
                    "spatial_transform": "perspective",
                    "occlusion_ratio": 0.8,
                },
            ),
            _step(3, "object_occlusion"),
        ),
        constraints=RecipeConstraints(
            max_steps=3,
            max_variants=2,
            max_storage_bytes=10,
            max_gpu_hours=0.0001,
            supported_spatial_transforms=("affine",),
            max_occlusion_ratio=0.5,
        ),
    )

    result = RecipeBuilder().validate(
        recipe,
        ATTACK_CATALOG,
        task="detection2d",
        model_capabilities=frozenset({"input_gradient", "detection_loss"}),
        annotation_types=frozenset({"boxes"}),
        modality="image",
        online=False,
        requested_variants=3,
        bytes_per_variant=20,
    )

    assert not result.valid
    assert {
        "duplicate_attack:object_occlusion",
        "forbidden_pair:fgsm+pgd",
        "multiple_white_box_steps",
        "recipe_length_exceeded",
        "unsupported_spatial_transform:perspective",
        "occlusion_ratio_exceeded",
        "variant_cap_exceeded",
        "storage_cap_exceeded",
        "gpu_cap_exceeded",
    } <= set(result.errors)


def test_random_builders_are_seeded_without_replacement_and_honor_filters() -> None:
    load_attacks()
    builder = RecipeBuilder()
    request = RandomNRequest(
        count=3,
        steps_per_recipe=2,
        seed=195,
        allowlist=("gaussian_noise", "contrast", "jpeg_compression", "pixelate"),
        blocklist=("pixelate",),
        required_attacks=("gaussian_noise",),
        severity_min=1,
        severity_max=3,
    )

    first = builder.random_n(request, ATTACK_CATALOG)
    second = builder.random_n(request, ATTACK_CATALOG)

    assert [recipe.recipe_hash for recipe in first] == [
        recipe.recipe_hash for recipe in second
    ]
    assert len({recipe.recipe_hash for recipe in first}) == 3
    for recipe in first:
        names = [step.attack_name for step in recipe.steps]
        assert len(names) == len(set(names))
        assert "gaussian_noise" in names
        assert "pixelate" not in names
        assert all(1 <= step.severity <= 3 for step in recipe.steps)


def test_stratified_and_sweep_builders_honor_quotas_and_seeds() -> None:
    load_attacks()
    builder = RecipeBuilder()
    stratified = builder.random_by_group(
        StratifiedRandomRequest(
            count=2,
            group_quotas={"A": 1, "C": 1},
            seed=9,
            allowlist=("gaussian_noise", "contrast", "random_erasing", "sensor_fault"),
        ),
        ATTACK_CATALOG,
    )
    sweep = builder.sweep(
        SweepRequest(
            attack_names=("gaussian_noise", "contrast"),
            severities=(1, 3),
            seed=9,
        ),
        ATTACK_CATALOG,
    )

    assert len(stratified) == 2
    assert all(len(recipe.steps) == 2 for recipe in stratified)
    assert len(sweep) == 4
    assert len({step.seed for recipe in sweep for step in recipe.steps}) == 4
