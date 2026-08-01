"""Patch loading, EOT transforms, placement, and regularisers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.core.hashing import array_digest
from src.core.image_ops import box_blur, clip01, nearest_resize
from src.core.types import Box, Sample
from src.datasets.io import load_image

_UNSET = object()


@dataclass(frozen=True, slots=True)
class EotTransform:
    scale: float
    rotation_degrees: float
    brightness: float
    blur_radius: int


@dataclass(frozen=True, slots=True)
class TransformedPatch:
    image: np.ndarray
    mask: np.ndarray
    transform: EotTransform


def load_patch(
    path: str | None,
    *,
    size: int = 32,
    expected_hash: str | None = None,
    required_algorithm: str | None = None,
    required_objective: str | None = None,
    required_source_label: str | None | object = _UNSET,
    required_target_label: str | None | object = _UNSET,
    allow_builtin: bool = False,
) -> tuple[np.ndarray, str]:
    if path is None:
        if not allow_builtin:
            raise ValueError("patch_path is required; train or provide a patch artifact")
        rows, cols = np.indices((size, size))
        checker = ((rows // 4 + cols // 4) % 2).astype(np.float32)
        patch = np.stack((checker, 1.0 - checker, np.full_like(checker, 0.5)), axis=2)
        artifact_hash = array_digest(patch, length=32)
        if expected_hash is not None and expected_hash != artifact_hash:
            raise ValueError("built-in patch does not match expected artifact hash")
        return patch, artifact_hash
    resolved = Path(path).expanduser().resolve()
    patch = load_image(resolved)
    artifact_hash = array_digest(patch, length=32)
    manifest_path = resolved.parent / "patch-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared_hash = manifest.get("artifact_hash")
        if declared_hash != artifact_hash:
            raise ValueError(
                f"patch artifact hash mismatch: {declared_hash!r} != {artifact_hash!r}"
            )
        if (
            required_algorithm is not None
            and manifest.get("algorithm") != required_algorithm
        ):
            raise ValueError(
                f"patch algorithm mismatch: {manifest.get('algorithm')!r} "
                f"!= {required_algorithm!r}"
            )
        config = manifest.get("config", {})
        declared_objective = manifest.get(
            "objective_kind",
            config.get("surrogate", {}).get("objective"),
        )
        if (
            required_objective is not None
            and declared_objective != required_objective
        ):
            raise ValueError(
                f"patch objective mismatch: {declared_objective!r} "
                f"!= {required_objective!r}"
            )
        declared_source_label = manifest.get(
            "placement_label",
            config.get("source_label"),
        )
        if (
            required_source_label is not _UNSET
            and declared_source_label != required_source_label
        ):
            raise ValueError(
                f"patch source label mismatch: {declared_source_label!r} "
                f"!= {required_source_label!r}"
            )
        declared_target_label = manifest.get(
            "objective_target_label",
            config.get("target_label")
            or config.get("surrogate", {}).get("target_label"),
        )
        if (
            required_target_label is not _UNSET
            and declared_target_label != required_target_label
        ):
            raise ValueError(
                f"patch target label mismatch: {declared_target_label!r} "
                f"!= {required_target_label!r}"
            )
    elif required_algorithm is not None:
        raise ValueError(f"patch artifact manifest does not exist: {manifest_path}")
    if expected_hash is not None and expected_hash != artifact_hash:
        raise ValueError(
            f"patch artifact does not match expected hash: "
            f"{expected_hash!r} != {artifact_hash!r}"
        )
    return patch, artifact_hash


def select_box(sample: Sample, target_label: str | None = None) -> Box:
    candidates = [
        box for box in sample.boxes if target_label is None or box.label == target_label
    ]
    if not candidates:
        raise ValueError(
            f"sample {sample.sample_id!r} has no box for target label {target_label!r}"
        )
    return max(candidates, key=lambda box: box.area)


def transform_patch(
    patch: np.ndarray,
    rng: np.random.Generator,
    *,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rotation_degrees: float = 20.0,
    brightness_delta: float = 0.2,
    blur_radius: int = 1,
) -> np.ndarray:
    transform = sample_eot_transform(
        rng,
        scale_range=scale_range,
        rotation_degrees=rotation_degrees,
        brightness_delta=brightness_delta,
        blur_radius=blur_radius,
    )
    return apply_eot_transform(patch, transform).image


def sample_eot_transform(
    rng: np.random.Generator,
    *,
    scale_range: tuple[float, float] = (0.8, 1.2),
    rotation_degrees: float = 20.0,
    brightness_delta: float = 0.2,
    blur_radius: int = 1,
) -> EotTransform:
    return EotTransform(
        scale=float(rng.uniform(*scale_range)),
        rotation_degrees=float(rng.uniform(-rotation_degrees, rotation_degrees)),
        brightness=float(
            rng.uniform(1.0 - brightness_delta, 1.0 + brightness_delta)
        ),
        blur_radius=(
            blur_radius
            if blur_radius > 0 and bool(rng.integers(0, 2))
            else 0
        ),
    )


def apply_eot_transform(
    patch: np.ndarray,
    transform: EotTransform,
) -> TransformedPatch:
    mask = np.ones(patch.shape[:2], dtype=np.float32)
    rotated = _rotate_nearest(patch, transform.rotation_degrees)
    rotated_mask = _rotate_nearest(
        mask[..., None],
        transform.rotation_degrees,
    )[..., 0]
    transformed = clip01(rotated * transform.brightness)
    if transform.blur_radius > 0:
        transformed = clip01(box_blur(transformed, transform.blur_radius))
    return TransformedPatch(
        image=transformed,
        mask=np.clip(rotated_mask, 0.0, 1.0).astype(np.float32),
        transform=transform,
    )


def inverse_eot_gradient(
    region_gradient: np.ndarray,
    transformed_mask: np.ndarray,
    transform: EotTransform,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Approximate the adjoint EOT map back to the trainable base patch."""
    region_mask = nearest_resize(
        transformed_mask[..., None],
        region_gradient.shape[0],
        region_gradient.shape[1],
    )
    masked = region_gradient * region_mask
    restored = nearest_resize(masked, output_shape[0], output_shape[1])
    if transform.blur_radius > 0:
        restored = box_blur(restored, transform.blur_radius)
    restored = restored * transform.brightness
    return _rotate_nearest(restored, -transform.rotation_degrees)


def place_patch(
    sample: Sample,
    patch: np.ndarray,
    box: Box,
    *,
    area_fraction: float,
    rng: np.random.Generator,
    random_offset: bool,
    mask: np.ndarray | None = None,
) -> tuple[Sample, tuple[slice, slice]]:
    height, width = sample.image.shape[:2]
    side = max(1, int(round(np.sqrt(max(1.0, box.area * area_fraction)))))
    side = min(side, max(1, int(box.width)), max(1, int(box.height)))
    resized = nearest_resize(patch, side, side)
    min_x = max(0, int(np.floor(box.x1)))
    min_y = max(0, int(np.floor(box.y1)))
    max_x = min(width - side, max(min_x, int(np.ceil(box.x2)) - side))
    max_y = min(height - side, max(min_y, int(np.ceil(box.y2)) - side))
    if random_offset:
        x1 = int(rng.integers(min_x, max_x + 1)) if max_x > min_x else min_x
        y1 = int(rng.integers(min_y, max_y + 1)) if max_y > min_y else min_y
    else:
        x1 = min(max_x, max(min_x, int(round((box.x1 + box.x2 - side) / 2))))
        y1 = min(max_y, max(min_y, int(round((box.y1 + box.y2 - side) / 2))))
    region = (slice(y1, y1 + side), slice(x1, x1 + side))
    image = sample.image.copy()
    if mask is None:
        image[region] = resized
    else:
        resized_mask = nearest_resize(mask[..., None], side, side)
        image[region] = (
            resized * resized_mask + image[region] * (1.0 - resized_mask)
        )
    return sample.with_image(clip01(image)), region


def total_variation(patch: np.ndarray) -> float:
    vertical = np.abs(np.diff(patch, axis=0)).mean() if patch.shape[0] > 1 else 0.0
    horizontal = np.abs(np.diff(patch, axis=1)).mean() if patch.shape[1] > 1 else 0.0
    return float(vertical + horizontal)


def nps_loss(patch: np.ndarray, palette: np.ndarray | None = None) -> float:
    if palette is None:
        levels = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        red, green, blue = np.meshgrid(levels, levels, levels, indexing="ij")
        colors = np.stack((red, green, blue), axis=-1).reshape(-1, 3)
    else:
        colors = palette
    distances = np.linalg.norm(patch[..., None, :] - colors[None, None, ...], axis=3)
    return float(np.min(distances, axis=2).mean())


def _rotate_nearest(image: np.ndarray, degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine, sine = float(np.cos(radians)), float(np.sin(radians))
    height, width = image.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    shifted_x, shifted_y = xx - center_x, yy - center_y
    source_x = cosine * shifted_x + sine * shifted_y + center_x
    source_y = -sine * shifted_x + cosine * shifted_y + center_y
    rounded_x = np.rint(source_x).astype(int)
    rounded_y = np.rint(source_y).astype(int)
    valid = (
        (rounded_x >= 0)
        & (rounded_x < width)
        & (rounded_y >= 0)
        & (rounded_y < height)
    )
    result = np.zeros_like(image, dtype=np.float32)
    result[valid] = image[rounded_y[valid], rounded_x[valid]]
    return result
