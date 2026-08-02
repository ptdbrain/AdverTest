"""Per-attack tests for ``random_erasing``: erased area, nesting, region count."""

from __future__ import annotations

import numpy as np
import pytest

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.types import Sample

SEVERITIES = (1, 2, 3, 4, 5)


def _flat_sample(size: int = 256) -> Sample:
    """Uniform grey frame, so an erased pixel is exactly a changed pixel."""
    return Sample("flat", np.full((size, size, 3), 0.5, dtype=np.float32))


def _erased_mask(attacked: Sample, sample: Sample) -> np.ndarray:
    return np.any(attacked.image != sample.image, axis=2)


def _run(severity: int, seed: int = 3, **params: object) -> tuple[Sample, Sample]:
    sample = _flat_sample()
    attack = get_attack("random_erasing", fill="black", **params)
    return sample, attack.run(sample, severity, AttackContext(rng=np.random.default_rng(seed)))


@pytest.mark.parametrize("severity", SEVERITIES)
def test_erased_area_matches_the_budget(severity: int) -> None:
    """Regions may overlap or clip at the border, so the budget is an upper bound."""
    sample, attacked = _run(severity)
    budget = get_attack("random_erasing").params.area_fraction_per_severity[severity - 1]
    erased = float(_erased_mask(attacked, sample).mean())
    assert erased <= budget * 1.05
    assert erased >= budget * 0.7


def test_masks_are_nested_across_severities() -> None:
    """Same seed, higher severity: the erased set only ever grows."""
    masks = [_erased_mask(*reversed(_run(severity))) for severity in SEVERITIES]
    for weaker, stronger in zip(masks, masks[1:], strict=False):
        assert np.array_equal(weaker & stronger, weaker)


def test_perturbation_grows_at_every_step() -> None:
    sample = _flat_sample()
    distances = [
        float(np.linalg.norm(_run(severity)[1].image - sample.image)) for severity in SEVERITIES
    ]
    assert distances == sorted(distances)


def test_single_region_is_one_rectangle() -> None:
    sample, attacked = _run(3, min_regions=1, max_regions=1)
    mask = _erased_mask(attacked, sample)
    rows, columns = np.nonzero(mask)
    bounding_area = (rows.max() - rows.min() + 1) * (columns.max() - columns.min() + 1)
    assert int(mask.sum()) == bounding_area


def test_black_fill_writes_zeros() -> None:
    sample, attacked = _run(5)
    assert float(attacked.image[_erased_mask(attacked, sample)].max()) == 0.0


def test_region_bounds_are_validated() -> None:
    with pytest.raises(ValueError, match="min_regions"):
        get_attack("random_erasing", min_regions=4, max_regions=2)
