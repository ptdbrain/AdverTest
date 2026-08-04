"""The reference dataset must be deterministic and contract-compliant."""

from __future__ import annotations

import pytest

from src.core.hashing import array_digest
from src.core.types import CLASSES, validate_image
from src.datasets import get_dataset
from src.datasets.base import AnonymizationRequiredError, DatasetSource


def test_same_seed_gives_identical_pixels() -> None:
    first = get_dataset("synthetic_shapes", n_samples=2, seed=11).load()
    second = get_dataset("synthetic_shapes", n_samples=2, seed=11).load()
    assert [array_digest(sample.image) for sample in first] == [
        array_digest(sample.image) for sample in second
    ]


def test_different_seed_gives_different_pixels() -> None:
    first = get_dataset("synthetic_shapes", n_samples=1, seed=11).load()[0]
    other = get_dataset("synthetic_shapes", n_samples=1, seed=12).load()[0]
    assert array_digest(first.image) != array_digest(other.image)


def test_samples_satisfy_the_image_contract() -> None:
    for sample in get_dataset("synthetic_shapes", n_samples=3, seed=5).load():
        validate_image(sample.image)
        assert sample.anonymized is True
        assert sample.depth is not None and sample.depth.shape == sample.image.shape[:2]


def test_ground_truth_is_inside_the_frame() -> None:
    for sample in get_dataset("synthetic_shapes", n_samples=3, seed=5).load():
        height, width = sample.image.shape[:2]
        assert sample.boxes, "every scene must contain at least one object"
        for box in sample.boxes:
            assert box.label in CLASSES
            assert 0 <= box.x1 < box.x2 <= width
            assert 0 <= box.y1 < box.y2 <= height


def test_limit_caps_the_batch() -> None:
    assert len(get_dataset("synthetic_shapes", n_samples=10, seed=5).load(limit=3)) == 3


def test_unknown_parameter_is_rejected() -> None:
    with pytest.raises(Exception, match="extra_inputs_are_not_permitted|Extra inputs"):
        get_dataset("synthetic_shapes", nsample=3)


def test_non_anonymised_dataset_cannot_be_evaluated() -> None:
    """Plan §6: the gate has no bypass, so the loader itself must refuse."""

    class RawFootage(DatasetSource):
        """Stand-in for a raw KITTI import."""

        name = "test_raw_footage"
        anonymized = False

        def load(self, limit: int | None = None):
            return []

    with pytest.raises(AnonymizationRequiredError, match="anonymis"):
        RawFootage().require_anonymized()
