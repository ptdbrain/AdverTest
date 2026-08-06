"""Six named recipes implemented without any preset-specific execution path."""

from __future__ import annotations

from src.attacks.recipes import AttackRecipe, AttackRecipeStep


def _step(
    position: int,
    attack_name: str,
    severity: int,
    seed: int,
    *,
    version: str = "1.0.0",
    cost: float = 1.0,
) -> AttackRecipeStep:
    return AttackRecipeStep(
        position=position,
        attack_name=attack_name,
        implementation_version=version,
        severity=severity,
        seed=seed,
        expected_cost=cost,
    )


PRESETS: dict[str, AttackRecipe] = {
    "low_visibility": AttackRecipe(
        name="Low Visibility",
        steps=(
            _step(0, "depth_fog", 3, 19501),
            _step(1, "contrast", 2, 19502),
        ),
    ),
    "wet_camera": AttackRecipe(
        name="Wet Camera",
        steps=(
            _step(0, "spatter", 3, 19511),
            _step(1, "gaussian_blur", 2, 19512),
        ),
    ),
    "poor_camera_pipeline": AttackRecipe(
        name="Poor Camera Pipeline",
        steps=(
            _step(0, "jpeg_compression", 3, 19521),
            _step(1, "pixelate", 2, 19522),
        ),
    ),
    "partial_obstruction": AttackRecipe(
        name="Partial Obstruction",
        steps=(_step(0, "object_occlusion", 3, 19531),),
    ),
    "adversarial_stress_yolo": AttackRecipe(
        name="Adversarial Stress YOLO",
        steps=(_step(0, "pgd", 3, 19541, cost=4.0),),
    ),
    "segmentation_boundary_stress_sam": AttackRecipe(
        name="Segmentation Boundary Stress SAM",
        steps=(_step(0, "sam2_pgd", 3, 19551, cost=4.0),),
    ),
}


def get_preset(name: str) -> AttackRecipe:
    try:
        return PRESETS[name]
    except KeyError:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(f"unknown recipe preset {name!r}; available: {available}") from None
