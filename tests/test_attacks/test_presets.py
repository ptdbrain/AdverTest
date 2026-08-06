from __future__ import annotations

from src.attacks.presets import PRESETS, get_preset
from src.attacks.recipes import AttackRecipe


def test_six_named_presets_resolve_to_ordinary_versioned_recipes() -> None:
    assert set(PRESETS) == {
        "low_visibility",
        "wet_camera",
        "poor_camera_pipeline",
        "partial_obstruction",
        "adversarial_stress_yolo",
        "segmentation_boundary_stress_sam",
    }
    assert all(isinstance(recipe, AttackRecipe) for recipe in PRESETS.values())
    assert all(recipe.catalog_version == "1.0.0" for recipe in PRESETS.values())
    assert all(recipe.recipe_hash for recipe in PRESETS.values())


def test_get_preset_returns_the_same_immutable_recipe() -> None:
    first = get_preset("low_visibility")
    second = get_preset("low_visibility")

    assert first is second
    assert [step.attack_name for step in first.steps] == ["depth_fog", "contrast"]
