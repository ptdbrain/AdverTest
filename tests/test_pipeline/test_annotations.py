from __future__ import annotations

import numpy as np
import pytest

from src.core.hashing import array_digest
from src.core.types import Box, Sample
from src.pipeline.annotations import (
    AnnotationPolicy,
    AnnotationTransformer,
    SpatialTransform,
)


def _sample() -> Sample:
    image = np.zeros((6, 8, 3), dtype=np.float32)
    image[1:5, 2:6] = 0.8
    mask = np.zeros((6, 8), dtype=np.bool_)
    mask[1:5, 2:6] = True
    instance = np.zeros((6, 8), dtype=np.bool_)
    instance[2:4, 3:5] = True
    return Sample(
        sample_id="sample-1",
        image=image,
        boxes=(Box(2, 1, 6, 5, "Car"),),
        mask=mask,
        anonymized=True,
        meta={
            "instance_masks": {"car-1": instance},
            "valid_region": np.ones((6, 8), dtype=np.bool_),
        },
    )


def test_identity_preserves_pixels_annotations_and_logs_hashes() -> None:
    source = _sample()

    transformed, log = AnnotationTransformer().apply(
        source,
        SpatialTransform.identity(source.image.shape[:2]),
        AnnotationPolicy(),
    )

    assert array_digest(transformed.image) == array_digest(source.image)
    assert transformed.boxes == source.boxes
    assert np.array_equal(transformed.mask, source.mask)
    assert log.objects[0].original_geometry == (2.0, 1.0, 6.0, 5.0)
    assert log.objects[0].transformed_geometry == (2.0, 1.0, 6.0, 5.0)
    assert log.masks[0].original_hash == log.masks[0].transformed_hash


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (SpatialTransform.crop(x=1, y=1, width=6, height=4), (1.0, 0.0, 5.0, 4.0)),
        (
            SpatialTransform.translate(
                dx=1,
                dy=0,
                source_shape=(6, 8),
            ),
            (3.0, 1.0, 7.0, 5.0),
        ),
        (
            SpatialTransform.scale(
                sx=0.5,
                sy=0.5,
                output_shape=(3, 4),
            ),
            (1.0, 0.5, 3.0, 2.5),
        ),
        (
            SpatialTransform.horizontal_flip(source_shape=(6, 8)),
            (2.0, 1.0, 6.0, 5.0),
        ),
        (
            SpatialTransform.rotate_90(source_shape=(6, 8), clockwise=True),
            (1.0, 2.0, 5.0, 6.0),
        ),
    ],
)
def test_box_geometry_uses_the_shared_matrix(
    transform: SpatialTransform,
    expected: tuple[float, float, float, float],
) -> None:
    transformed, _ = AnnotationTransformer().apply(
        _sample(),
        transform,
        AnnotationPolicy(),
    )

    assert transformed.boxes[0].as_tuple() == pytest.approx(expected)


def test_mask_transform_is_nearest_neighbor_and_preserves_instance_ids() -> None:
    source = _sample()
    transform = SpatialTransform.resize_pad(
        source_shape=(6, 8),
        output_shape=(12, 12),
    )

    transformed, log = AnnotationTransformer().apply(
        source,
        transform,
        AnnotationPolicy(),
    )

    assert transformed.mask is not None
    assert transformed.mask.dtype == np.bool_
    assert set(np.unique(transformed.mask)) <= {False, True}
    assert set(transformed.meta["instance_masks"]) == {"car-1"}
    assert log.masks[1].instance_id == "car-1"
    assert transformed.meta["valid_region"].dtype == np.bool_


def test_clipping_and_visible_ratio_obey_explicit_drop_policy() -> None:
    source = _sample()
    transform = SpatialTransform.crop(x=4, y=0, width=4, height=6)
    keep_policy = AnnotationPolicy(
        min_visible_ratio=0.75,
        drop_boxes_below_visible_ratio=False,
    )
    drop_policy = AnnotationPolicy(
        min_visible_ratio=0.75,
        drop_boxes_below_visible_ratio=True,
    )

    kept, keep_log = AnnotationTransformer().apply(source, transform, keep_policy)
    dropped, drop_log = AnnotationTransformer().apply(source, transform, drop_policy)

    assert kept.boxes[0].as_tuple() == (0.0, 1.0, 2.0, 5.0)
    assert keep_log.objects[0].visible_ratio == pytest.approx(0.5)
    assert dropped.boxes == ()
    assert drop_log.objects[0].kept is False
    assert drop_log.objects[0].reason == "below_visible_ratio"


def test_affine_rotation_and_resize_pad_do_not_mutate_source() -> None:
    source = _sample()
    before = source.image.copy()
    transform = SpatialTransform.affine_rotation(
        angle_degrees=15,
        source_shape=source.image.shape[:2],
    )

    rotated, _ = AnnotationTransformer().apply(
        source,
        transform,
        AnnotationPolicy(),
    )

    assert rotated.image.shape == source.image.shape
    assert np.array_equal(source.image, before)
    assert rotated is not source


def test_occlusion_keeps_geometry_unless_policy_explicitly_allows_drop() -> None:
    source = _sample()
    occlusion = np.zeros((6, 8), dtype=np.bool_)
    occlusion[:, 2:6] = True
    transform = SpatialTransform.occlusion(occlusion)

    kept, keep_log = AnnotationTransformer().apply(
        source,
        transform,
        AnnotationPolicy(occlusion_may_drop=False, min_visible_ratio=0.5),
    )
    dropped, drop_log = AnnotationTransformer().apply(
        source,
        transform,
        AnnotationPolicy(occlusion_may_drop=True, min_visible_ratio=0.5),
    )

    assert kept.boxes == source.boxes
    assert keep_log.objects[0].visible_ratio == 0.0
    assert dropped.boxes == ()
    assert drop_log.objects[0].reason == "occluded_below_visible_ratio"
    assert np.array_equal(source.mask, kept.mask)


def test_empty_transformed_masks_are_logged_and_optionally_dropped() -> None:
    source = _sample()
    transformed, log = AnnotationTransformer().apply(
        source,
        SpatialTransform.translate(dx=20, dy=0, source_shape=(6, 8)),
        AnnotationPolicy(drop_empty_masks=True),
    )

    assert transformed.mask is None
    assert log.masks[0].kept is False
    assert log.masks[0].reason == "empty_after_transform"
