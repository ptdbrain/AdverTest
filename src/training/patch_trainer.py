"""Train reusable DPatch/Thys artifacts without running robustness metrics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.adapters import get_adapter
from src.attacks.adversarial._iterative import input_gradient
from src.attacks.patch._utils import (
    apply_eot_transform,
    inverse_eot_gradient,
    nps_loss,
    place_patch,
    sample_eot_transform,
    select_box,
    total_variation,
)
from src.core.hashing import array_digest, file_digest, stable_digest
from src.core.image_ops import clip01, nearest_resize
from src.core.objectives import AttackObjective
from src.datasets import get_dataset
from src.datasets.base import DatasetSource
from src.pipeline.generator import SurrogateConfig


class PatchTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str | None = None
    dataset_params: dict[str, Any] = Field(default_factory=dict)
    input_dir: str | None = None
    input_format: Literal["advertest", "kitti"] = "advertest"
    anonymization_manifest: str | None = None
    surrogate: SurrogateConfig
    algorithm: Literal["dpatch", "thys_patch"] = "dpatch"
    source_label: str | None = None
    target_label: str | None = None
    patch_size: int = Field(default=64, ge=8, le=512)
    area_fraction: float = Field(default=0.15, gt=0.0, le=1.0)
    iterations: int = Field(default=500, ge=1)
    learning_rate: float = Field(default=0.03, gt=0.0)
    tv_weight: float = Field(default=2.5, ge=0.0)
    nps_weight: float = Field(default=0.01, ge=0.0)
    eot: bool = True
    seed: int = 20260730
    train_limit: int = Field(default=200, ge=1)
    sample_ids: list[str] | None = None
    output_dir: str = "data/patches"

    @model_validator(mode="after")
    def validate_source(self) -> PatchTrainingConfig:
        if (self.dataset_name is None) == (self.input_dir is None):
            raise ValueError("provide exactly one of dataset_name or input_dir")
        if (
            self.surrogate.objective == "targeted"
            and self.target_label is None
            and self.surrogate.target_label is None
        ):
            raise ValueError("targeted patch training requires target_label")
        if self.sample_ids is not None:
            if not self.sample_ids:
                raise ValueError("sample_ids must not be empty")
            if len(self.sample_ids) != len(set(self.sample_ids)):
                raise ValueError("sample_ids must not contain duplicates")
        return self


@dataclass(frozen=True, slots=True)
class PatchArtifact:
    artifact_id: str
    root: Path
    patch_path: Path
    preview_path: Path
    manifest_path: Path
    artifact_hash: str

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "root": str(self.root),
            "patch_path": str(self.patch_path),
            "preview_path": str(self.preview_path),
            "manifest_path": str(self.manifest_path),
            "artifact_hash": self.artifact_hash,
        }


class PatchTrainer:
    """Optimise one universal patch using surrogate input gradients."""

    version: ClassVar[str] = "2.0.0"

    def train(self, config: PatchTrainingConfig) -> PatchArtifact:
        source = self._source(config)
        source.require_anonymized()
        samples = source.load(None if config.sample_ids is not None else config.train_limit)
        if config.sample_ids is not None:
            by_id = {sample.sample_id: sample for sample in samples}
            missing = sorted(set(config.sample_ids) - by_id.keys())
            if missing:
                raise ValueError(f"patch training sample_ids do not exist: {missing}")
            samples = [by_id[sample_id] for sample_id in config.sample_ids]
        if not samples:
            raise ValueError("patch training source returned no samples")
        placement_label = (
            "Pedestrian"
            if config.algorithm == "thys_patch"
            else config.source_label
        )
        objective_target = (
            "Pedestrian"
            if config.algorithm == "thys_patch"
            else config.target_label or config.surrogate.target_label
        )
        if placement_label is not None:
            samples = [
                sample
                for sample in samples
                if any(box.label == placement_label for box in sample.boxes)
            ]
            if not samples:
                raise ValueError(
                    f"patch training source has no boxes for {placement_label!r}"
                )
        if (
            config.surrogate.name in {"yolo11", "faster_rcnn", "sam2_surrogate"}
            and config.surrogate.checkpoint is None
        ):
            raise ValueError(
                f"surrogate {config.surrogate.name!r} requires an explicit checkpoint path"
            )
        checkpoint_hash: str | None = None
        if config.surrogate.checkpoint is not None:
            checkpoint = Path(config.surrogate.checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"surrogate checkpoint does not exist: {checkpoint}"
                )
            checkpoint_hash = file_digest(checkpoint)
        model_params = dict(config.surrogate.params)
        if config.surrogate.checkpoint is not None:
            model_params.setdefault("weights", config.surrogate.checkpoint)
        model_params.setdefault("device", config.surrogate.device)
        try:
            model = get_adapter(config.surrogate.name, **model_params)
        except TypeError:
            model_params.pop("device", None)
            model = get_adapter(config.surrogate.name, **model_params)
        for capability in ("input_gradient", "detection_loss"):
            if not model.supports(capability):  # type: ignore[arg-type]
                raise ValueError(
                    f"patch training requires surrogate capability {capability!r}"
                )

        rng = np.random.default_rng(config.seed)
        patch = rng.random((config.patch_size, config.patch_size, 3), dtype=np.float32)
        objective = AttackObjective(
            kind="vanishing" if config.algorithm == "thys_patch" else config.surrogate.objective,
            target_label=objective_target,
        )
        history: list[dict[str, float | int]] = []
        used_ids: set[str] = set()
        for iteration in range(config.iterations):
            sample = samples[iteration % len(samples)]
            used_ids.add(sample.sample_id)
            box = select_box(sample, placement_label)
            transformed_patch = patch
            transformed_mask = None
            transform = None
            area_fraction = config.area_fraction
            if config.eot:
                transformed = apply_eot_transform(
                    patch,
                    sample_eot_transform(rng),
                )
                transformed_patch = transformed.image
                transformed_mask = transformed.mask
                transform = transformed.transform
                area_fraction *= transform.scale**2
            attacked, region = place_patch(
                sample,
                transformed_patch,
                box,
                area_fraction=min(1.0, area_fraction),
                rng=rng,
                random_offset=config.eot,
                mask=transformed_mask,
            )
            gradient = input_gradient(model, attacked, objective)
            region_gradient = gradient[region]
            if transform is not None and transformed_mask is not None:
                patch_gradient = inverse_eot_gradient(
                    region_gradient,
                    transformed_mask,
                    transform,
                    (config.patch_size, config.patch_size),
                )
            else:
                patch_gradient = nearest_resize(
                    region_gradient,
                    config.patch_size,
                    config.patch_size,
                )
            tv = total_variation(patch)
            nps = nps_loss(patch)
            regularizer = _regularizer_gradient(patch, config.tv_weight, config.nps_weight)
            patch = clip01(
                patch + config.learning_rate * np.sign(patch_gradient - regularizer)
            )
            if iteration == 0 or (iteration + 1) % max(1, config.iterations // 20) == 0:
                history.append(
                    {
                        "iteration": iteration + 1,
                        "attack_loss": model.loss_for_attack(attacked, objective),
                        "tv": tv,
                        "nps": nps,
                    }
                )

        payload = config.model_dump(mode="json")
        source_fingerprint = stable_digest(
            [
                {
                    "sample_id": sample.sample_id,
                    "image_hash": array_digest(sample.image, length=32),
                    "mask_hash": (
                        array_digest(sample.mask, length=32)
                        if sample.mask is not None
                        else None
                    ),
                    "boxes": [
                        (*box.as_tuple(), box.label)
                        for box in sample.boxes
                    ],
                }
                for sample in samples
            ],
            length=32,
        )
        artifact_id = stable_digest(
            {
                "config": payload,
                "samples": sorted(used_ids),
                "source_fingerprint": source_fingerprint,
                "surrogate": model.metadata().version,
                "checkpoint_hash": checkpoint_hash,
                "trainer_version": self.version,
            },
            length=16,
        )
        root = Path(config.output_dir).expanduser().resolve() / config.algorithm / artifact_id
        root.mkdir(parents=True, exist_ok=True)
        patch_path = root / "patch.npy"
        preview_path = root / "patch.png"
        _write_npy(patch_path, patch)
        _write_png(preview_path, patch)
        artifact_hash = array_digest(patch, length=32)
        manifest = {
            "format": "advertest-patch-v1",
            "artifact_id": artifact_id,
            "algorithm": config.algorithm,
            "trainer_version": self.version,
            "artifact_hash": artifact_hash,
            "surrogate": model.metadata().name,
            "surrogate_version": model.metadata().version,
            "checkpoint_hash": checkpoint_hash,
            "config": payload,
            "objective_kind": objective.kind,
            "placement_label": placement_label,
            "objective_target_label": objective_target,
            "train_sample_ids": sorted(used_ids),
            "training_source_fingerprint": source_fingerprint,
            "loss_history": history,
            "patch_path": patch_path.name,
            "preview_path": preview_path.name,
        }
        manifest_path = root / "patch-manifest.json"
        _write_json(manifest_path, manifest)
        return PatchArtifact(
            artifact_id,
            root,
            patch_path,
            preview_path,
            manifest_path,
            artifact_hash,
        )

    @staticmethod
    def _source(config: PatchTrainingConfig) -> DatasetSource:
        if config.dataset_name is not None:
            return get_dataset(config.dataset_name, **config.dataset_params)
        return get_dataset(
            "folder_dataset",
            root=config.input_dir,
            input_format=config.input_format,
            anonymization_manifest=config.anonymization_manifest,
        )


def _regularizer_gradient(
    patch: np.ndarray,
    tv_weight: float,
    nps_weight: float,
) -> np.ndarray:
    tv_gradient = np.zeros_like(patch)
    tv_gradient[:-1] += np.sign(patch[:-1] - patch[1:])
    tv_gradient[1:] += np.sign(patch[1:] - patch[:-1])
    tv_gradient[:, :-1] += np.sign(patch[:, :-1] - patch[:, 1:])
    tv_gradient[:, 1:] += np.sign(patch[:, 1:] - patch[:, :-1])
    printable = np.rint(patch * 7.0) / 7.0
    nps_gradient = patch - printable
    return tv_weight * tv_gradient + nps_weight * nps_gradient


def _write_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def _write_png(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("patch preview requires Pillow: pip install Pillow") from exc
    temporary = path.with_name(f".{path.name}.tmp")
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
