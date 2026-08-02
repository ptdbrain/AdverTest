"""Per-attack tests for ``frame_freeze``: drift ladder, zoom, shape."""

from __future__ import annotations

import numpy as np
import pytest

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.types import Sample

SEVERITIES = (1, 2, 3, 4, 5)


def _structured_sample(size: int = 128) -> Sample:
    """Bright square on a dark frame: an ego-motion warp has to move something."""
    image = np.full((size, size, 3), 0.15, dtype=np.float32)
    image[40:80, 40:80] = 0.85
    return Sample("square", image)


def _run(severity: int, sample: Sample, seed: int = 2, **params: object) -> Sample:
    attack = get_attack("frame_freeze", **params)
    return attack.run(sample, severity, AttackContext(rng=np.random.default_rng(seed)))


def test_perturbation_grows_at_every_step() -> None:
    """More stale frames means more accumulated ego motion (sanity check #2)."""
    sample = _structured_sample()
    distances = [
        float(np.linalg.norm(_run(severity, sample).image - sample.image))
        for severity in SEVERITIES
    ]
    assert distances == sorted(distances)


@pytest.mark.parametrize("severity", SEVERITIES)
def test_shape_and_range_are_preserved(severity: int) -> None:
    sample = _structured_sample()
    attacked = _run(severity, sample)
    assert attacked.image.shape == sample.image.shape
    assert attacked.image.dtype == np.float32
    assert 0.0 <= float(attacked.image.min()) and float(attacked.image.max()) <= 1.0


def test_forward_motion_makes_the_object_bigger() -> None:
    """Zoom about the centre: the bright square covers more pixels than before."""
    sample = _structured_sample()
    attacked = _run(5, sample, shift_px_per_frame=0.0, zoom_per_frame=0.05)
    before = int((sample.image[..., 0] > 0.5).sum())
    after = int((attacked.image[..., 0] > 0.5).sum())
    assert after > before


def test_lateral_drift_moves_the_object_sideways() -> None:
    sample = _structured_sample()
    attacked = _run(5, sample, zoom_per_frame=0.001, shift_px_per_frame=6.0)
    columns_before = np.nonzero(np.any(sample.image[..., 0] > 0.5, axis=0))[0]
    columns_after = np.nonzero(np.any(attacked.image[..., 0] > 0.5, axis=0))[0]
    assert int(columns_before.min()) != int(columns_after.min())


def test_stale_frame_count_is_configurable() -> None:
    sample = _structured_sample()
    mild = _run(1, sample, stale_frames_per_severity=(1,))
    severe = _run(1, sample, stale_frames_per_severity=(10,))
    assert np.linalg.norm(severe.image - sample.image) > np.linalg.norm(mild.image - sample.image)
