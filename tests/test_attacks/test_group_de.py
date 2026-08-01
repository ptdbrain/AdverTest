from __future__ import annotations

import numpy as np
import pytest

from src.adapters.base import ModelAdapter
from src.attacks import get_attack
from src.attacks.adversarial._iterative import input_gradient, iterative_linf
from src.attacks.base import AttackContext, ModelRequiredError
from src.attacks.patch._utils import (
    EotTransform,
    apply_eot_transform,
    inverse_eot_gradient,
    nps_loss,
    place_patch,
    total_variation,
    transform_patch,
)
from src.core.objectives import AttackObjective
from src.core.types import Sample


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("pgd", {"steps": 2}),
        ("mi_fgsm", {"steps": 2}),
        ("cw_l2", {"iterations": 2, "binary_search_steps": 1}),
        ("tog", {"steps": 2}),
        ("dag", {"iterations": 2}),
        ("sam2_pgd", {"steps": 2}),
    ],
)
def test_group_d_plugins_produce_valid_bounded_variants(
    name: str,
    params: dict[str, int],
    adapter: ModelAdapter,
    sample: Sample,
) -> None:
    attack = get_attack(name, **params)
    context = AttackContext(rng=np.random.default_rng(17), model=adapter)
    attacked = attack.run(sample, 1, context)
    assert attacked.image.shape == sample.image.shape
    assert attacked.image.dtype == np.float32
    assert attacked.boxes == sample.boxes
    assert np.array_equal(attacked.mask, sample.mask)
    assert not np.array_equal(attacked.image, sample.image)


def test_pgd_respects_linf_budget(
    adapter: ModelAdapter,
    sample: Sample,
) -> None:
    epsilon = 3 / 255
    attack = get_attack(
        "pgd",
        epsilon_per_severity=(epsilon,),
        steps=4,
        restarts=2,
    )
    attacked = attack.run(
        sample,
        1,
        AttackContext(rng=np.random.default_rng(3), model=adapter),
    )
    assert float(np.max(np.abs(attacked.image - sample.image))) <= epsilon + 1e-6


def test_cw_respects_l2_budget(
    adapter: ModelAdapter,
    sample: Sample,
) -> None:
    radius = 0.2
    attacked = get_attack(
        "cw_l2",
        radius_per_severity=(radius,),
        iterations=2,
        binary_search_steps=1,
    ).run(
        sample,
        1,
        AttackContext(rng=np.random.default_rng(3), model=adapter),
    )
    assert float(np.linalg.norm(attacked.image - sample.image)) <= radius + 1e-6


def test_reference_gradient_is_finite_and_nonzero(
    adapter: ModelAdapter,
    sample: Sample,
) -> None:
    gradient = input_gradient(adapter, sample, AttackObjective())
    assert np.isfinite(gradient).all()
    assert np.any(np.abs(gradient) > 0)


def test_iterative_attack_stops_after_gradient_saturates(
    adapter: ModelAdapter,
    sample: Sample,
) -> None:
    calls = 0
    original_input_gradient = adapter.input_gradient

    def saturating_gradient(current, objective=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_input_gradient(current, objective)
        return np.zeros_like(current.image)

    adapter.input_gradient = saturating_gradient  # type: ignore[method-assign]
    attacked = iterative_linf(
        sample,
        AttackContext(rng=np.random.default_rng(2), model=adapter),
        epsilon=2 / 255,
        steps=3,
        step_size=1 / 255,
        random_start=False,
    )

    assert calls == 2
    assert not np.array_equal(attacked.image, sample.image)


@pytest.mark.parametrize(
    ("name", "params", "capability"),
    [
        ("cw_l2", {"iterations": 1, "binary_search_steps": 1}, "class_margin"),
        ("tog", {"steps": 1}, "class_logits"),
        ("dag", {"iterations": 1}, "dense_proposals"),
        ("sam2_pgd", {"steps": 1}, "segmentation_loss"),
    ],
)
def test_specialized_attack_rejects_missing_capability(
    name: str,
    params: dict[str, int],
    capability: str,
    sample: Sample,
) -> None:
    class NoCapabilityAdapter(ModelAdapter):
        name = "none"

        def predict(self, samples):
            return []

        def metadata(self):
            raise NotImplementedError

    with pytest.raises(ModelRequiredError, match=capability):
        get_attack(name, **params).run(
            sample,
            1,
            AttackContext(rng=np.random.default_rng(0), model=NoCapabilityAdapter()),
        )


@pytest.mark.parametrize("name", ["dpatch", "thys_patch"])
def test_patch_plugins_are_deterministic(
    name: str,
    sample: Sample,
) -> None:
    params = (
        {"target_label": None, "allow_builtin_patch": True}
        if name == "dpatch"
        else {
            "target_label": sample.boxes[0].label,
            "allow_builtin_patch": True,
        }
    )
    attack = get_attack(name, **params)
    first = attack.run(sample, 1, AttackContext(rng=np.random.default_rng(9)))
    second = attack.run(sample, 1, AttackContext(rng=np.random.default_rng(9)))
    assert np.array_equal(first.image, second.image)
    assert first.boxes == sample.boxes


def test_patch_area_stays_inside_selected_box(sample: Sample) -> None:
    box = sample.boxes[0]
    patch = np.ones((16, 16, 3), dtype=np.float32)
    _, region = place_patch(
        sample,
        patch,
        box,
        area_fraction=0.10,
        rng=np.random.default_rng(4),
        random_offset=True,
    )
    y_slice, x_slice = region
    assert int(box.x1) <= x_slice.start < x_slice.stop <= int(np.ceil(box.x2))
    assert int(box.y1) <= y_slice.start < y_slice.stop <= int(np.ceil(box.y2))
    actual_area = (x_slice.stop - x_slice.start) * (y_slice.stop - y_slice.start)
    expected_area = box.area * 0.10
    assert abs(actual_area - expected_area) <= max(box.width, box.height)


def test_eot_and_patch_regularizers_are_reproducible_and_finite() -> None:
    patch = np.linspace(0.0, 1.0, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    first = transform_patch(patch, np.random.default_rng(8))
    second = transform_patch(patch, np.random.default_rng(8))
    assert np.array_equal(first, second)
    assert np.isfinite(total_variation(first))
    assert np.isfinite(nps_loss(first))


def test_eot_scale_changes_physical_patch_area(sample: Sample) -> None:
    patch = np.ones((16, 16, 3), dtype=np.float32)
    box = sample.boxes[0]
    small = apply_eot_transform(
        patch,
        EotTransform(0.8, 0.0, 1.0, 0),
    )
    large = apply_eot_transform(
        patch,
        EotTransform(1.2, 0.0, 1.0, 0),
    )
    _, small_region = place_patch(
        sample,
        small.image,
        box,
        area_fraction=0.10 * small.transform.scale**2,
        rng=np.random.default_rng(1),
        random_offset=False,
        mask=small.mask,
    )
    _, large_region = place_patch(
        sample,
        large.image,
        box,
        area_fraction=0.10 * large.transform.scale**2,
        rng=np.random.default_rng(1),
        random_offset=False,
        mask=large.mask,
    )

    small_area = (small_region[0].stop - small_region[0].start) ** 2
    large_area = (large_region[0].stop - large_region[0].start) ** 2
    assert large_area > small_area


def test_eot_rotation_has_transparent_corners_and_invertible_gradient() -> None:
    patch = np.ones((16, 16, 3), dtype=np.float32)
    transform = EotTransform(1.0, 20.0, 0.9, 1)
    transformed = apply_eot_transform(patch, transform)
    gradient = inverse_eot_gradient(
        np.ones((24, 24, 3), dtype=np.float32),
        transformed.mask,
        transform,
        patch.shape[:2],
    )

    assert transformed.mask[0, 0] == 0.0
    assert transformed.mask[-1, -1] == 0.0
    assert gradient.shape == patch.shape
    assert np.isfinite(gradient).all()
    assert np.any(np.abs(gradient) > 0)
