"""Shared spatial transforms for images, boxes, masks, and valid regions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from src.core.hashing import array_digest, stable_digest
from src.core.types import Box, Sample


class AnnotationPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    version: str = "1.0.0"
    min_visible_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    drop_boxes_below_visible_ratio: bool = False
    drop_empty_masks: bool = False
    occlusion_may_drop: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class SpatialTransform:
    """One source-to-output homogeneous matrix plus an output image shape."""

    matrix: np.ndarray
    output_shape: tuple[int, int]
    kind: str = "affine"
    version: str = "1.0.0"
    occlusion_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"transform matrix must be 3x3, got {matrix.shape}")
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
            raise ValueError("transform matrix must be finite and invertible")
        if self.output_shape[0] <= 0 or self.output_shape[1] <= 0:
            raise ValueError("output shape must be positive")
        normalized = np.ascontiguousarray(matrix)
        normalized.setflags(write=False)
        object.__setattr__(self, "matrix", normalized)
        if self.occlusion_mask is not None:
            occlusion = np.asarray(self.occlusion_mask)
            if occlusion.shape != self.output_shape or occlusion.dtype != np.bool_:
                raise ValueError("occlusion mask must be boolean and match output shape")
            immutable = np.ascontiguousarray(occlusion)
            immutable.setflags(write=False)
            object.__setattr__(self, "occlusion_mask", immutable)

    @classmethod
    def identity(cls, source_shape: tuple[int, int]) -> SpatialTransform:
        return cls(np.eye(3), source_shape, kind="identity")

    @classmethod
    def crop(
        cls,
        *,
        x: float,
        y: float,
        width: int,
        height: int,
    ) -> SpatialTransform:
        return cls(
            np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]]),
            (height, width),
            kind="crop",
        )

    @classmethod
    def translate(
        cls,
        *,
        dx: float,
        dy: float,
        source_shape: tuple[int, int],
    ) -> SpatialTransform:
        return cls(
            np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]]),
            source_shape,
            kind="translate",
        )

    @classmethod
    def scale(
        cls,
        *,
        sx: float,
        sy: float,
        output_shape: tuple[int, int],
    ) -> SpatialTransform:
        if sx <= 0 or sy <= 0:
            raise ValueError("scale factors must be positive")
        return cls(
            np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]),
            output_shape,
            kind="scale",
        )

    @classmethod
    def horizontal_flip(
        cls,
        *,
        source_shape: tuple[int, int],
    ) -> SpatialTransform:
        _, width = source_shape
        return cls(
            np.array([[-1.0, 0.0, width], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            source_shape,
            kind="horizontal_flip",
        )

    @classmethod
    def rotate_90(
        cls,
        *,
        source_shape: tuple[int, int],
        clockwise: bool,
    ) -> SpatialTransform:
        height, width = source_shape
        matrix = (
            np.array([[0.0, -1.0, height], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            if clockwise
            else np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, width], [0.0, 0.0, 1.0]])
        )
        return cls(matrix, (width, height), kind="rotate_90")

    @classmethod
    def affine_rotation(
        cls,
        *,
        angle_degrees: float,
        source_shape: tuple[int, int],
    ) -> SpatialTransform:
        height, width = source_shape
        radians = np.deg2rad(angle_degrees)
        cosine = float(np.cos(radians))
        sine = float(np.sin(radians))
        center_x, center_y = width / 2.0, height / 2.0
        matrix = np.array(
            [
                [
                    cosine,
                    -sine,
                    center_x - cosine * center_x + sine * center_y,
                ],
                [
                    sine,
                    cosine,
                    center_y - sine * center_x - cosine * center_y,
                ],
                [0.0, 0.0, 1.0],
            ]
        )
        return cls(matrix, source_shape, kind="affine_rotation")

    @classmethod
    def resize_pad(
        cls,
        *,
        source_shape: tuple[int, int],
        output_shape: tuple[int, int],
    ) -> SpatialTransform:
        source_height, source_width = source_shape
        output_height, output_width = output_shape
        scale = min(output_width / source_width, output_height / source_height)
        offset_x = (output_width - source_width * scale) / 2.0
        offset_y = (output_height - source_height * scale) / 2.0
        return cls(
            np.array(
                [
                    [scale, 0.0, offset_x],
                    [0.0, scale, offset_y],
                    [0.0, 0.0, 1.0],
                ]
            ),
            output_shape,
            kind="resize_pad",
        )

    @classmethod
    def occlusion(cls, mask: np.ndarray) -> SpatialTransform:
        return cls(
            np.eye(3),
            mask.shape,
            kind="occlusion",
            occlusion_mask=mask,
        )


class ObjectTransformRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    object_id: str
    original_geometry: tuple[float, float, float, float]
    transformed_geometry: tuple[float, float, float, float] | None
    original_hash: str
    transformed_hash: str | None
    visible_ratio: float = Field(ge=0.0, le=1.0)
    kept: bool
    reason: str
    policy_version: str


class MaskTransformRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    instance_id: str
    original_shape: tuple[int, int]
    transformed_shape: tuple[int, int] | None
    original_hash: str
    transformed_hash: str | None
    visible_ratio: float = Field(ge=0.0, le=1.0)
    kept: bool
    reason: str
    policy_version: str


class AnnotationTransformLog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    transform_kind: str
    transform_version: str
    matrix: tuple[tuple[float, float, float], ...]
    output_shape: tuple[int, int]
    policy_version: str
    source_image_hash: str
    transformed_image_hash: str
    objects: tuple[ObjectTransformRecord, ...] = ()
    masks: tuple[MaskTransformRecord, ...] = ()


class AnnotationTransformer:
    def apply(
        self,
        sample: Sample,
        transform: SpatialTransform,
        policy: AnnotationPolicy,
    ) -> tuple[Sample, AnnotationTransformLog]:
        if transform.occlusion_mask is not None:
            return self._apply_occlusion(sample, transform, policy)
        image = _warp_image(sample.image, transform)
        boxes, object_logs = _transform_boxes(sample.boxes, transform, policy)
        mask, semantic_log = _transform_optional_mask(
            sample.mask,
            "semantic",
            transform,
            policy,
        )
        metadata = dict(sample.meta)
        mask_logs: list[MaskTransformRecord] = []
        if semantic_log is not None:
            mask_logs.append(semantic_log)
        instance_masks = sample.meta.get("instance_masks")
        if isinstance(instance_masks, dict):
            transformed_instances: dict[str, np.ndarray] = {}
            for instance_id, instance_mask in instance_masks.items():
                transformed_mask, mask_log = _transform_optional_mask(
                    instance_mask,
                    str(instance_id),
                    transform,
                    policy,
                )
                if transformed_mask is not None:
                    transformed_instances[str(instance_id)] = transformed_mask
                if mask_log is not None:
                    mask_logs.append(mask_log)
            metadata["instance_masks"] = transformed_instances
        valid_region = sample.meta.get("valid_region")
        if isinstance(valid_region, np.ndarray):
            metadata["valid_region"] = _warp_mask(valid_region, transform)
        transformed = replace(
            sample,
            image=image,
            boxes=boxes,
            mask=mask,
            meta=metadata,
        )
        return transformed, _log(
            sample,
            transformed,
            transform,
            policy,
            object_logs,
            mask_logs,
        )

    def _apply_occlusion(
        self,
        sample: Sample,
        transform: SpatialTransform,
        policy: AnnotationPolicy,
    ) -> tuple[Sample, AnnotationTransformLog]:
        occlusion = transform.occlusion_mask
        assert occlusion is not None
        if occlusion.shape != sample.image.shape[:2]:
            raise ValueError("occlusion mask must match source image shape")
        image = sample.image.copy()
        image[occlusion] = 0.0
        kept_boxes: list[Box] = []
        object_logs: list[ObjectTransformRecord] = []
        for index, box in enumerate(sample.boxes):
            visible_ratio = _box_visible_under_occlusion(box, occlusion)
            drop = (
                policy.occlusion_may_drop
                and visible_ratio < policy.min_visible_ratio
            )
            if not drop:
                kept_boxes.append(box)
            geometry = box.as_tuple()
            object_logs.append(
                ObjectTransformRecord(
                    object_id=f"box-{index}",
                    original_geometry=geometry,
                    transformed_geometry=None if drop else geometry,
                    original_hash=_box_hash(box),
                    transformed_hash=None if drop else _box_hash(box),
                    visible_ratio=visible_ratio,
                    kept=not drop,
                    reason=(
                        "occluded_below_visible_ratio"
                        if drop
                        else "occlusion_gt_preserved"
                    ),
                    policy_version=policy.version,
                )
            )
        mask_logs = _occlusion_mask_logs(sample, occlusion, policy)
        transformed = replace(
            sample,
            image=image.astype(np.float32, copy=False),
            boxes=tuple(kept_boxes),
            meta=dict(sample.meta),
        )
        return transformed, _log(
            sample,
            transformed,
            transform,
            policy,
            object_logs,
            mask_logs,
        )


def _warp_image(image: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    inverse = np.linalg.inv(transform.matrix)
    coefficients = tuple(float(value) for value in inverse[:2].reshape(-1))
    output_height, output_width = transform.output_shape
    channels = [
        np.asarray(
            Image.fromarray(image[..., channel], mode="F").transform(
                (output_width, output_height),
                Image.Transform.AFFINE,
                coefficients,
                resample=Image.Resampling.BILINEAR,
                fillcolor=0.0,
            ),
            dtype=np.float32,
        )
        for channel in range(image.shape[2])
    ]
    return np.clip(np.stack(channels, axis=2), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def _warp_mask(mask: np.ndarray, transform: SpatialTransform) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("annotation mask must be a 2D numpy array")
    inverse = np.linalg.inv(transform.matrix)
    coefficients = tuple(float(value) for value in inverse[:2].reshape(-1))
    output_height, output_width = transform.output_shape
    warped = Image.fromarray(mask.astype(np.uint8, copy=False) * 255).transform(
        (output_width, output_height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return np.asarray(warped, dtype=np.uint8) > 0


def _transform_boxes(
    boxes: tuple[Box, ...],
    transform: SpatialTransform,
    policy: AnnotationPolicy,
) -> tuple[tuple[Box, ...], list[ObjectTransformRecord]]:
    output_height, output_width = transform.output_shape
    transformed_boxes: list[Box] = []
    logs: list[ObjectTransformRecord] = []
    for index, box in enumerate(boxes):
        corners = np.array(
            [
                [box.x1, box.y1, 1.0],
                [box.x2, box.y1, 1.0],
                [box.x2, box.y2, 1.0],
                [box.x1, box.y2, 1.0],
            ]
        )
        mapped = (transform.matrix @ corners.T).T
        mapped = mapped[:, :2] / mapped[:, 2:3]
        raw = (
            float(mapped[:, 0].min()),
            float(mapped[:, 1].min()),
            float(mapped[:, 0].max()),
            float(mapped[:, 1].max()),
        )
        clipped = (
            float(np.clip(raw[0], 0.0, output_width)),
            float(np.clip(raw[1], 0.0, output_height)),
            float(np.clip(raw[2], 0.0, output_width)),
            float(np.clip(raw[3], 0.0, output_height)),
        )
        raw_area = max(0.0, raw[2] - raw[0]) * max(0.0, raw[3] - raw[1])
        clipped_area = max(0.0, clipped[2] - clipped[0]) * max(
            0.0,
            clipped[3] - clipped[1],
        )
        visible_ratio = clipped_area / raw_area if raw_area > 0 else 0.0
        outside = clipped_area <= 0.0
        below = visible_ratio < policy.min_visible_ratio
        drop = outside or (policy.drop_boxes_below_visible_ratio and below)
        reason = (
            "outside_output"
            if outside
            else "below_visible_ratio"
            if drop
            else "kept"
        )
        transformed_box = None
        if not drop:
            transformed_box = Box(
                x1=clipped[0],
                y1=clipped[1],
                x2=clipped[2],
                y2=clipped[3],
                label=box.label,
                score=box.score,
            )
            transformed_boxes.append(transformed_box)
        logs.append(
            ObjectTransformRecord(
                object_id=f"box-{index}",
                original_geometry=box.as_tuple(),
                transformed_geometry=(
                    transformed_box.as_tuple() if transformed_box is not None else None
                ),
                original_hash=_box_hash(box),
                transformed_hash=(
                    _box_hash(transformed_box)
                    if transformed_box is not None
                    else None
                ),
                visible_ratio=visible_ratio,
                kept=not drop,
                reason=reason,
                policy_version=policy.version,
            )
        )
    return tuple(transformed_boxes), logs


def _transform_optional_mask(
    mask: np.ndarray | None,
    instance_id: str,
    transform: SpatialTransform,
    policy: AnnotationPolicy,
) -> tuple[np.ndarray | None, MaskTransformRecord | None]:
    if mask is None:
        return None, None
    original = np.asarray(mask, dtype=np.bool_)
    transformed = _warp_mask(original, transform)
    original_area = int(original.sum())
    visible_ratio = (
        min(1.0, float(transformed.sum()) / original_area)
        if original_area > 0
        else 0.0
    )
    empty = not bool(transformed.any())
    drop = empty and policy.drop_empty_masks
    return (
        None if drop else transformed,
        MaskTransformRecord(
            instance_id=instance_id,
            original_shape=original.shape,
            transformed_shape=None if drop else transformed.shape,
            original_hash=array_digest(original),
            transformed_hash=None if drop else array_digest(transformed),
            visible_ratio=visible_ratio,
            kept=not drop,
            reason="empty_after_transform" if drop else "kept",
            policy_version=policy.version,
        ),
    )


def _occlusion_mask_logs(
    sample: Sample,
    occlusion: np.ndarray,
    policy: AnnotationPolicy,
) -> list[MaskTransformRecord]:
    masks: list[tuple[str, np.ndarray]] = []
    if sample.mask is not None:
        masks.append(("semantic", np.asarray(sample.mask, dtype=np.bool_)))
    instance_masks = sample.meta.get("instance_masks")
    if isinstance(instance_masks, dict):
        masks.extend(
            (str(instance_id), np.asarray(mask, dtype=np.bool_))
            for instance_id, mask in instance_masks.items()
        )
    records: list[MaskTransformRecord] = []
    for instance_id, mask in masks:
        area = int(mask.sum())
        visible = int(np.logical_and(mask, ~occlusion).sum())
        ratio = visible / area if area else 0.0
        records.append(
            MaskTransformRecord(
                instance_id=instance_id,
                original_shape=mask.shape,
                transformed_shape=mask.shape,
                original_hash=array_digest(mask),
                transformed_hash=array_digest(mask),
                visible_ratio=ratio,
                kept=True,
                reason="occlusion_gt_preserved",
                policy_version=policy.version,
            )
        )
    return records


def _box_visible_under_occlusion(box: Box, occlusion: np.ndarray) -> float:
    height, width = occlusion.shape
    x1 = max(0, min(width, int(np.floor(box.x1))))
    y1 = max(0, min(height, int(np.floor(box.y1))))
    x2 = max(0, min(width, int(np.ceil(box.x2))))
    y2 = max(0, min(height, int(np.ceil(box.y2))))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((~occlusion[y1:y2, x1:x2]).mean())


def _box_hash(box: Box) -> str:
    return stable_digest(
        {
            "xyxy": box.as_tuple(),
            "label": box.label,
            "score": box.score,
        },
        length=64,
    )


def _log(
    source: Sample,
    transformed: Sample,
    transform: SpatialTransform,
    policy: AnnotationPolicy,
    object_logs: list[ObjectTransformRecord],
    mask_logs: list[MaskTransformRecord],
) -> AnnotationTransformLog:
    return AnnotationTransformLog(
        transform_kind=transform.kind,
        transform_version=transform.version,
        matrix=tuple(
            tuple(float(value) for value in row) for row in transform.matrix
        ),
        output_shape=transform.output_shape,
        policy_version=policy.version,
        source_image_hash=array_digest(source.image),
        transformed_image_hash=array_digest(transformed.image),
        objects=tuple(object_logs),
        masks=tuple(mask_logs),
    )
