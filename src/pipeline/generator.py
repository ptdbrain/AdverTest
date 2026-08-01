"""Offline attack-dataset generation without evaluation or model comparison."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.adapters import get_adapter
from src.adapters.base import ModelAdapter
from src.attacks import get_attack
from src.attacks.base import AttackContext, BaseAttack
from src.core.hashing import array_digest, file_digest, stable_digest
from src.core.objectives import AttackObjective, ObjectiveKind
from src.core.types import Sample, validate_image
from src.datasets import get_dataset
from src.datasets.base import DatasetSource
from src.datasets.io import boxes_payload, load_image, load_mask


class SurrogateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    checkpoint: str | None = None
    device: str = "cpu"
    objective: ObjectiveKind = "untargeted"
    target_label: str | None = None
    target_box_index: int | None = None


class AttackGenerationConfig(BaseModel):
    """One immutable generation recipe: one source, one attack, many severities."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str | None = None
    dataset_params: dict[str, Any] = Field(default_factory=dict)
    input_dir: str | None = None
    input_format: Literal["advertest", "kitti"] = "advertest"
    anonymization_manifest: str | None = None
    attack_name: str
    attack_params: dict[str, Any] = Field(default_factory=dict)
    severities: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5], min_length=1)
    seed: int = 20260730
    surrogate: SurrogateConfig | None = None
    output_dir: str = "data/attacked"
    preview: bool = True
    limit: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_source(self) -> AttackGenerationConfig:
        if (self.dataset_name is None) == (self.input_dir is None):
            raise ValueError("provide exactly one of dataset_name or input_dir")
        if len(set(self.severities)) != len(self.severities):
            raise ValueError("severities must not contain duplicates")
        if any(level < 0 for level in self.severities):
            raise ValueError("severities must be non-negative")
        return self


@dataclass(frozen=True, slots=True)
class GenerationReport:
    generation_id: str
    root: Path
    source: str
    attack: str
    n_source_samples: int
    n_variants: int
    resumed_variants: int
    estimated_canonical_bytes: int
    estimated_gradient_steps: int
    status: str = "complete"

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "root": str(self.root),
            "source": self.source,
            "attack": self.attack,
            "n_source_samples": self.n_source_samples,
            "n_variants": self.n_variants,
            "resumed_variants": self.resumed_variants,
            "estimated_canonical_bytes": self.estimated_canonical_bytes,
            "estimated_gradient_steps": self.estimated_gradient_steps,
            "status": self.status,
        }


class AttackDatasetGenerator:
    """Create a reloadable attack dataset and provenance manifest."""

    def generate(self, config: AttackGenerationConfig) -> GenerationReport:
        source = self._source(config)
        source.require_anonymized()
        attack = get_attack(config.attack_name, **config.attack_params)
        effective_limit = config.limit
        if effective_limit is None and attack.name == "cw_l2":
            effective_limit = 100
        samples = source.load(effective_limit)
        if not samples:
            raise ValueError("source dataset returned no samples")
        surrogate = self._surrogate(config, attack)
        objective = self._objective(config)
        self._preflight(attack, samples, surrogate, objective, config.severities)
        source_name = config.dataset_name or Path(config.input_dir or "").name
        source_fingerprint = _dataset_fingerprint(samples)
        surrogate_provenance = _surrogate_provenance(config, surrogate)
        estimate = _estimate_generation(samples, attack, config.severities)
        generation_id = _generation_id(
            config,
            source_name,
            source_fingerprint,
            attack,
            surrogate_provenance,
        )
        root = Path(config.output_dir).expanduser().resolve() / source_name / attack.name / generation_id
        root.mkdir(parents=True, exist_ok=True)
        for name in ("images", "labels", "masks", "artifacts"):
            (root / name).mkdir(exist_ok=True)
        if config.preview:
            (root / "previews").mkdir(exist_ok=True)

        config_payload = config.model_dump(mode="json")
        config_path = root / "config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != config_payload:
                raise ValueError(f"generation directory contains a different config: {root}")
        else:
            _write_json(config_path, config_payload)
        artifact_path = _persist_attack_artifact(root, attack)

        records = _read_manifest(root / "manifest.jsonl")
        by_variant = {record["variant_id"]: record for record in records}
        resumed = 0
        descriptor = self._descriptor(
            config,
            generation_id,
            source_name,
            len(samples),
            source_fingerprint,
            estimate,
            status="in_progress",
        )
        _write_json(root / "dataset.json", descriptor)

        try:
            for sample in samples:
                for severity in config.severities:
                    variant_id = _variant_id(
                        config,
                        sample,
                        attack,
                        severity,
                        surrogate_provenance,
                    )
                    existing = by_variant.get(variant_id)
                    if existing is not None and _record_is_valid(root, existing):
                        resumed += 1
                        continue
                    attacked, derived_seed = self._apply(
                        attack,
                        sample,
                        severity,
                        config.seed,
                        surrogate,
                        objective,
                    )
                    record = self._persist(
                        root,
                        sample,
                        attacked,
                        attack,
                        severity,
                        derived_seed,
                        variant_id,
                        surrogate_provenance,
                        artifact_path,
                        config.preview,
                    )
                    by_variant[variant_id] = record
                    _write_manifest(root / "manifest.jsonl", list(by_variant.values()))
        except Exception as exc:
            descriptor["status"] = "incomplete"
            descriptor["error"] = f"{type(exc).__name__}: {exc}"
            descriptor["n_variants"] = len(by_variant)
            _write_json(root / "dataset.json", descriptor)
            raise

        descriptor["status"] = "complete"
        descriptor["n_variants"] = len(by_variant)
        descriptor["manifest_hash"] = stable_digest(
            sorted(by_variant.values(), key=lambda row: row["variant_id"]),
            length=32,
        )
        _write_json(root / "dataset.json", descriptor)
        return GenerationReport(
            generation_id=generation_id,
            root=root,
            source=source_name,
            attack=attack.name,
            n_source_samples=len(samples),
            n_variants=len(by_variant),
            resumed_variants=resumed,
            estimated_canonical_bytes=estimate["canonical_bytes"],
            estimated_gradient_steps=estimate["gradient_steps"],
        )

    @staticmethod
    def _source(config: AttackGenerationConfig) -> DatasetSource:
        if config.dataset_name is not None:
            return get_dataset(config.dataset_name, **config.dataset_params)
        return get_dataset(
            "folder_dataset",
            root=config.input_dir,
            input_format=config.input_format,
            anonymization_manifest=config.anonymization_manifest,
        )

    @staticmethod
    def _surrogate(
        config: AttackGenerationConfig,
        attack: BaseAttack,
    ) -> ModelAdapter | None:
        if not attack.needs_model:
            return None
        if config.surrogate is None:
            raise ValueError(f"attack {attack.name!r} requires a surrogate config")
        if (
            config.surrogate.name in {"yolo11", "faster_rcnn", "sam2_surrogate"}
            and config.surrogate.checkpoint is None
        ):
            raise ValueError(
                f"surrogate {config.surrogate.name!r} requires an explicit checkpoint path"
            )
        params = dict(config.surrogate.params)
        if config.surrogate.checkpoint is not None:
            params.setdefault("weights", config.surrogate.checkpoint)
        params.setdefault("device", config.surrogate.device)
        try:
            return get_adapter(config.surrogate.name, **params)
        except TypeError:
            params.pop("device", None)
            return get_adapter(config.surrogate.name, **params)

    @staticmethod
    def _objective(config: AttackGenerationConfig) -> AttackObjective:
        if config.surrogate is None:
            return AttackObjective()
        return AttackObjective(
            kind=config.surrogate.objective,
            target_label=config.surrogate.target_label,
            target_box_index=config.surrogate.target_box_index,
        )

    @staticmethod
    def _preflight(
        attack: BaseAttack,
        samples: list[Sample],
        surrogate: ModelAdapter | None,
        objective: AttackObjective,
        severities: list[int],
    ) -> None:
        sample_ids = [sample.sample_id for sample in samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("source dataset contains duplicate sample IDs")
        invalid_severities = [
            severity
            for severity in severities
            if not 0 <= severity <= attack.severity_levels
        ]
        if invalid_severities:
            raise ValueError(
                f"severity for {attack.name!r} must be 0..{attack.severity_levels}, "
                f"got {invalid_severities}"
            )
        if objective.kind == "targeted" and objective.target_label is None:
            raise ValueError("targeted objective requires surrogate.target_label")
        for sample in samples:
            attack.validate_requirements(sample, surrogate)
            if objective.kind == "targeted" and not sample.boxes:
                raise ValueError(
                    f"targeted objective requires boxes; sample {sample.sample_id!r} has none"
                )
            if (
                objective.target_box_index is not None
                and not 0 <= objective.target_box_index < len(sample.boxes)
            ):
                raise ValueError(
                    f"target_box_index {objective.target_box_index} is invalid for "
                    f"sample {sample.sample_id!r}"
                )

    @staticmethod
    def _apply(
        attack: BaseAttack,
        sample: Sample,
        severity: int,
        run_seed: int,
        surrogate: ModelAdapter | None,
        objective: AttackObjective,
    ) -> tuple[Sample, int]:
        seed_material = stable_digest(
            {
                "seed": run_seed,
                "sample": sample.sample_id,
                "attack": attack.name,
                "severity": severity,
            }
        )
        derived_seed = int(seed_material, 16) % (2**32)
        context = AttackContext(
            rng=np.random.default_rng(derived_seed),
            model=surrogate,
            objective=objective,
        )
        attacked = attack.run(sample, severity, context)
        validate_image(attacked.image, like=sample.image)
        if attacked.boxes != sample.boxes:
            raise ValueError(f"attack {attack.name!r} modified ground-truth boxes")
        if not _same_optional_array(attacked.mask, sample.mask):
            raise ValueError(f"attack {attack.name!r} modified the ground-truth mask")
        _validate_perturbation_budget(attack, sample, attacked, severity)
        return attacked, derived_seed

    @staticmethod
    def _persist(
        root: Path,
        source: Sample,
        attacked: Sample,
        attack: BaseAttack,
        severity: int,
        seed: int,
        variant_id: str,
        surrogate: dict[str, str | None],
        artifact_path: Path | None,
        preview: bool,
    ) -> dict[str, Any]:
        image_rel = Path("images") / f"{variant_id}.npy"
        label_rel = Path("labels") / f"{variant_id}.json"
        _write_npy(root / image_rel, attacked.image)
        _write_json(root / label_rel, boxes_payload(attacked.boxes))
        mask_rel: Path | None = None
        if attacked.mask is not None:
            mask_rel = Path("masks") / f"{variant_id}.npy"
            _write_npy(root / mask_rel, attacked.mask)
        preview_rel: Path | None = None
        if preview:
            preview_rel = Path("previews") / f"{variant_id}.png"
            _write_png(root / preview_rel, attacked.image)
        delta = attacked.image.astype(np.float64) - source.image.astype(np.float64)
        label_hash = stable_digest(boxes_payload(attacked.boxes), length=32)
        mask_hash = (
            array_digest(attacked.mask, length=32)
            if attacked.mask is not None
            else None
        )
        return {
            "variant_id": variant_id,
            "source_sample_id": source.sample_id,
            "source_hash": array_digest(source.image, length=32),
            "source_sample_hash": _sample_digest(source),
            "output_hash": array_digest(attacked.image, length=32),
            "attack": attack.name,
            "attack_version": attack.version,
            "attack_class": f"{type(attack).__module__}.{type(attack).__name__}",
            "attack_params": attack.param_dict(),
            "severity": severity,
            "seed": seed,
            "linf": float(np.max(np.abs(delta))),
            "l2": float(np.linalg.norm(delta)),
            "surrogate": surrogate["name"],
            "surrogate_version": surrogate["version"],
            "checkpoint_hash": surrogate["checkpoint_hash"],
            "patch_artifact_hash": getattr(attack, "artifact_hash", None),
            "patch_artifact_path": (
                artifact_path.as_posix()
                if artifact_path is not None
                else None
            ),
            "image_path": image_rel.as_posix(),
            "preview_path": preview_rel.as_posix() if preview_rel else None,
            "label_path": label_rel.as_posix(),
            "label_hash": label_hash,
            "mask_path": mask_rel.as_posix() if mask_rel else None,
            "mask_hash": mask_hash,
            "anonymized": source.anonymized,
        }

    @staticmethod
    def _descriptor(
        config: AttackGenerationConfig,
        generation_id: str,
        source: str,
        n_source_samples: int,
        source_fingerprint: str,
        estimate: dict[str, int],
        *,
        status: str,
    ) -> dict[str, Any]:
        return {
            "format": "advertest-generated-v1",
            "generation_id": generation_id,
            "source": source,
            "source_fingerprint": source_fingerprint,
            "attack": config.attack_name,
            "status": status,
            "anonymized": True,
            "n_source_samples": n_source_samples,
            "n_variants": 0,
            "canonical_image_format": "npy-float32-hwc-0-1",
            "estimate": estimate,
        }


def inspect_generated_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    descriptor = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    records = _read_manifest(root / "manifest.jsonl")
    invalid = [record["variant_id"] for record in records if not _record_is_valid(root, record)]
    ordered_records = sorted(records, key=lambda row: row["variant_id"])
    actual_manifest_hash = stable_digest(ordered_records, length=32)
    count_matches = descriptor.get("n_variants") == len(records)
    manifest_hash_matches = descriptor.get("manifest_hash") == actual_manifest_hash
    return {
        **descriptor,
        "root": str(root),
        "manifest_records": len(records),
        "invalid_variants": invalid,
        "count_matches": count_matches,
        "manifest_hash_matches": manifest_hash_matches,
        "valid": (
            descriptor.get("status") == "complete"
            and not invalid
            and count_matches
            and manifest_hash_matches
        ),
    }


def _generation_id(
    config: AttackGenerationConfig,
    source_name: str,
    source_fingerprint: str,
    attack: BaseAttack,
    surrogate: dict[str, str | None],
) -> str:
    payload = config.model_dump(mode="json", exclude={"output_dir"})
    return stable_digest(
        {
            "source": source_name,
            "source_fingerprint": source_fingerprint,
            "resolved_attack": {
                "class": f"{type(attack).__module__}.{type(attack).__qualname__}",
                "version": attack.version,
                "params": attack.param_dict(),
            },
            "resolved_surrogate": surrogate,
            **payload,
        },
        length=16,
    )


def _variant_id(
    config: AttackGenerationConfig,
    sample: Sample,
    attack: BaseAttack,
    severity: int,
    surrogate: dict[str, str | None],
) -> str:
    return stable_digest(
        {
            "sample": sample.sample_id,
            "source_sample_hash": _sample_digest(sample),
            "attack": attack.name,
            "params": attack.param_dict(),
            "severity": severity,
            "seed": config.seed,
            "surrogate": surrogate,
        },
        length=20,
    )


def _same_optional_array(first: np.ndarray | None, second: np.ndarray | None) -> bool:
    if first is None or second is None:
        return first is second
    return bool(np.array_equal(first, second))


def _sample_digest(sample: Sample) -> str:
    return stable_digest(
        {
            "sample_id": sample.sample_id,
            "image_hash": array_digest(sample.image, length=32),
            "labels": boxes_payload(sample.boxes),
            "mask_hash": (
                array_digest(sample.mask, length=32)
                if sample.mask is not None
                else None
            ),
        },
        length=32,
    )


def _dataset_fingerprint(samples: list[Sample]) -> str:
    return stable_digest(
        [_sample_digest(sample) for sample in samples],
        length=32,
    )


def _surrogate_provenance(
    config: AttackGenerationConfig,
    surrogate: ModelAdapter | None,
) -> dict[str, str | None]:
    if surrogate is None:
        return {"name": None, "version": None, "checkpoint_hash": None}
    checkpoint_hash: str | None = None
    if config.surrogate and config.surrogate.checkpoint:
        checkpoint = Path(config.surrogate.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"surrogate checkpoint does not exist: {checkpoint}")
        checkpoint_hash = file_digest(checkpoint)
    metadata = surrogate.metadata()
    return {
        "name": metadata.name,
        "version": metadata.version,
        "checkpoint_hash": checkpoint_hash,
    }


def _estimate_generation(
    samples: list[Sample],
    attack: BaseAttack,
    severities: list[int],
) -> dict[str, int]:
    variants = len(samples) * len(severities)
    bytes_per_severity = sum(
        sample.image.nbytes
        + len(json.dumps(boxes_payload(sample.boxes), sort_keys=True).encode("utf-8"))
        + (sample.mask.nbytes if sample.mask is not None else 0)
        for sample in samples
    )
    params = attack.param_dict()
    steps_per_variant = 0
    if attack.needs_gradients:
        if attack.name == "cw_l2":
            steps_per_variant = int(params.get("iterations", 1)) * int(
                params.get("binary_search_steps", 1)
            )
        else:
            steps_per_variant = int(params.get("steps", params.get("iterations", 1)))
            steps_per_variant *= int(params.get("restarts", 1))
    active_variants = len(samples) * sum(severity > 0 for severity in severities)
    return {
        "variants": variants,
        "canonical_bytes": bytes_per_severity * len(severities),
        "gradient_steps": active_variants * steps_per_variant,
    }


def _validate_perturbation_budget(
    attack: BaseAttack,
    source: Sample,
    attacked: Sample,
    severity: int,
) -> None:
    if severity == 0:
        return
    params = attack.param_dict()
    delta = attacked.image.astype(np.float64) - source.image.astype(np.float64)
    epsilon_values = params.get("epsilon_per_severity")
    if epsilon_values:
        budget = attack.level(severity, epsilon_values)
        actual = float(np.max(np.abs(delta)))
        if actual > budget + 1e-6:
            raise ValueError(
                f"attack {attack.name!r} exceeded L-inf budget: "
                f"{actual:.8f} > {budget:.8f}"
            )
    radius_values = params.get("radius_per_severity")
    if radius_values:
        budget = attack.level(severity, radius_values)
        actual = float(np.linalg.norm(delta))
        if actual > budget + 1e-6:
            raise ValueError(
                f"attack {attack.name!r} exceeded L2 budget: "
                f"{actual:.8f} > {budget:.8f}"
            )


def _persist_attack_artifact(root: Path, attack: BaseAttack) -> Path | None:
    if attack.generation_mode != "artifact":
        return None
    patch = getattr(attack, "patch", None)
    artifact_hash = getattr(attack, "artifact_hash", None)
    if not isinstance(patch, np.ndarray) or not isinstance(artifact_hash, str):
        raise ValueError(
            f"artifact attack {attack.name!r} did not expose patch and artifact_hash"
        )
    actual_hash = array_digest(patch, length=32)
    if actual_hash != artifact_hash:
        raise ValueError(
            f"artifact attack {attack.name!r} has an invalid in-memory patch hash"
        )
    relative = Path("artifacts") / f"{attack.name}-{artifact_hash}.npy"
    destination = root / relative
    if destination.is_file():
        try:
            existing = np.load(destination, allow_pickle=False)
        except (OSError, ValueError):
            existing = None
        if existing is not None and array_digest(existing, length=32) == artifact_hash:
            return relative
    _write_npy(destination, patch)
    return relative


def _record_is_valid(root: Path, record: dict[str, Any]) -> bool:
    image_path = root / record["image_path"]
    if not image_path.is_file():
        return False
    try:
        image = np.load(image_path, allow_pickle=False)
        validate_image(image)
    except (OSError, TypeError, ValueError):
        return False
    if array_digest(image, length=32) != record.get("output_hash"):
        return False
    label_path = root / record.get("label_path", "")
    if not label_path.is_file():
        return False
    try:
        label_payload = json.loads(label_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if stable_digest(label_payload, length=32) != record.get("label_hash"):
        return False
    mask_path_value = record.get("mask_path")
    if mask_path_value:
        mask_path = root / mask_path_value
        try:
            mask = load_mask(mask_path)
        except (OSError, ValueError):
            return False
        if mask is None or array_digest(mask, length=32) != record.get("mask_hash"):
            return False
    preview_path_value = record.get("preview_path")
    if preview_path_value:
        try:
            preview = load_image(root / preview_path_value)
        except (OSError, ValueError):
            return False
        if preview.shape != image.shape:
            return False
    artifact_path_value = record.get("patch_artifact_path")
    if artifact_path_value:
        try:
            artifact = np.load(root / artifact_path_value, allow_pickle=False)
        except (OSError, ValueError):
            return False
        if array_digest(artifact, length=32) != record.get("patch_artifact_hash"):
            return False
    return True


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in sorted(records, key=lambda row: row["variant_id"])
    )
    _atomic_write(path, content.encode("utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(path, content.encode("utf-8"))


def _write_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def _write_png(path: Path, image: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PNG previews require Pillow: pip install Pillow") from exc
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    temporary = path.with_name(f".{path.name}.tmp")
    Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
