"""Per-attack tests for ``sensor_fault``: dead-pixel count, nesting, bands."""

from __future__ import annotations

import numpy as np
import pytest

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.types import Sample

SEVERITIES = (1, 2, 3, 4, 5)
SIZE = 200


def _flat_sample() -> Sample:
    return Sample("flat", np.full((SIZE, SIZE, 3), 0.5, dtype=np.float32))


def _run(severity: int, seed: int = 11, **params: object) -> tuple[Sample, Sample]:
    sample = _flat_sample()
    attack = get_attack("sensor_fault", **params)
    return sample, attack.run(sample, severity, AttackContext(rng=np.random.default_rng(seed)))


def _changed(attacked: Sample, sample: Sample) -> np.ndarray:
    return np.any(attacked.image != sample.image, axis=2)


@pytest.mark.parametrize("severity", SEVERITIES)
def test_dead_pixel_count_matches_the_configured_fraction(severity: int) -> None:
    """Bands are disabled here so the mask is only the stuck pixels."""
    sample, attacked = _run(severity, n_bands_per_severity=(0,))
    fraction = get_attack("sensor_fault").params.dead_pixel_fraction_per_severity[severity - 1]
    assert float(_changed(attacked, sample).mean()) == pytest.approx(fraction, rel=0.02)


def test_at_least_one_pixel_dies_even_at_an_absurdly_small_fraction() -> None:
    """Severity 1 on a small frame must never be a silent no-op."""
    sample, attacked = _run(1, dead_pixel_fraction_per_severity=(1e-9,), n_bands_per_severity=(0,))
    assert int(_changed(attacked, sample).sum()) == 1


def test_dead_masks_are_nested_across_severities() -> None:
    masks = [
        _changed(*reversed(_run(severity, n_bands_per_severity=(0,)))) for severity in SEVERITIES
    ]
    for weaker, stronger in zip(masks, masks[1:], strict=False):
        assert np.array_equal(weaker & stronger, weaker)


def test_band_count_follows_the_severity_ladder() -> None:
    """Bands are the only source of fully blacked-out rows."""
    band_height = 8
    for severity, expected in zip(SEVERITIES, (0, 1, 2, 3, 4), strict=True):
        sample, attacked = _run(
            severity,
            dead_pixel_fraction_per_severity=(1e-9,),
            band_height_px=band_height,
        )
        full_rows = int(np.all(_changed(attacked, sample), axis=1).sum())
        assert full_rows <= expected * band_height
        if expected:
            assert full_rows >= band_height


def test_perturbation_grows_at_every_step() -> None:
    sample = _flat_sample()
    distances = [
        float(np.linalg.norm(_run(severity)[1].image - sample.image)) for severity in SEVERITIES
    ]
    assert distances == sorted(distances)


def test_black_fill_writes_zeros() -> None:
    sample, attacked = _run(5)
    assert float(attacked.image[_changed(attacked, sample)].max()) == 0.0
