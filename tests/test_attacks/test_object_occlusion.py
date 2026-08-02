"""Per-attack tests for ``object_occlusion``: coverage ratio, anchor, fallback."""

from __future__ import annotations

import numpy as np
import pytest

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.types import Box, Sample

SEVERITIES = (1, 2, 3, 4, 5)
#: One box, well inside a grey frame, so coverage is measurable exactly.
BOX = Box(20.0, 30.0, 60.0, 110.0, "Car")


def _sample(*boxes: Box) -> Sample:
    return Sample("flat", np.full((128, 128, 3), 0.5, dtype=np.float32), boxes)


def _run(severity: int, sample: Sample, seed: int = 5, **params: object) -> Sample:
    attack = get_attack("object_occlusion", fill="black", **params)
    return attack.run(sample, severity, AttackContext(rng=np.random.default_rng(seed)))


def _covered(attacked: Sample, sample: Sample, box: Box) -> float:
    """Fraction of ``box`` whose pixels the attack changed."""
    rows = slice(int(box.y1), int(box.y2))
    columns = slice(int(box.x1), int(box.x2))
    region = np.any(attacked.image[rows, columns] != sample.image[rows, columns], axis=2)
    return float(region.mean())


@pytest.mark.parametrize("severity", SEVERITIES)
def test_covered_fraction_matches_the_configured_ratio(severity: int) -> None:
    sample = _sample(BOX)
    ratio = get_attack("object_occlusion").params.cover_ratio_per_severity[severity - 1]
    assert _covered(_run(severity, sample), sample, BOX) == pytest.approx(ratio, abs=0.02)


def test_bottom_anchor_starts_at_the_bottom_edge_of_the_box() -> None:
    sample = _sample(BOX)
    attacked = _run(2, sample)
    changed_rows = np.nonzero(np.any(attacked.image != sample.image, axis=(1, 2)))[0]
    assert int(changed_rows.max()) == int(BOX.y2) - 1


def test_every_box_is_occluded() -> None:
    second = Box(80.0, 10.0, 120.0, 50.0, "Pedestrian")
    sample = _sample(BOX, second)
    attacked = _run(3, sample)
    assert _covered(attacked, sample, BOX) > 0.0
    assert _covered(attacked, sample, second) > 0.0


def test_max_objects_limits_the_targets() -> None:
    second = Box(80.0, 10.0, 120.0, 50.0, "Pedestrian")
    sample = _sample(BOX, second)
    attacked = _run(3, sample, max_objects=1)
    assert _covered(attacked, sample, BOX) > 0.0
    assert _covered(attacked, sample, second) == 0.0


def test_frame_without_ground_truth_is_rejected() -> None:
    sample = _sample()
    with pytest.raises(ValueError, match="requires annotations: boxes"):
        _run(3, sample)


def test_perturbation_grows_at_every_step() -> None:
    sample = _sample(BOX)
    distances = [
        float(np.linalg.norm(_run(severity, sample).image - sample.image))
        for severity in SEVERITIES
    ]
    assert distances == sorted(distances)


def test_ground_truth_boxes_are_not_modified() -> None:
    sample = _sample(BOX)
    assert _run(5, sample).boxes == sample.boxes
