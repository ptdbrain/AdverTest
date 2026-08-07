"""Deterministic, leakage-safe defense-training dataset manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.attacks.catalog import CATALOG_ENTRIES
from src.core.hashing import array_digest, stable_digest
from src.datasets.contracts import (
    DatasetVersion,
    GeneratedVariantRecord,
    SplitManifest,
)
from src.datasets.splits import SplitBuilder
from src.training.contracts import DefenseProfile


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class GeneratedTrainingSource(_FrozenContract):
    root: str
    records: tuple[GeneratedVariantRecord, ...]


class TrainingDatasetConfig(_FrozenContract):
    base_version: DatasetVersion
    split_manifest: SplitManifest
    defense_profile: DefenseProfile
    generated_sources: tuple[GeneratedTrainingSource, ...] = ()
    output_dir: str
    storage_budget_bytes: int = Field(gt=0)
    schema_version: Literal["1.0.0"] = "1.0.0"


class TrainingDatasetEstimate(_FrozenContract):
    clean_count: int
    generated_count: int
    estimated_bytes: int
    online_count: int
    offline_count: int
    class_distribution: dict[str, int]
    object_size_distribution: dict[str, int]
    recipe_distribution: dict[str, int]
    severity_distribution: dict[int, int]
    warnings: tuple[str, ...]
    hard_cap_violations: tuple[str, ...]


class TrainingManifestEntry(_FrozenContract):
    entry_id: str
    kind: Literal["clean", "generated", "hard_example"]
    source_sample_id: str
    source_dataset_version_id: str
    artifact_path: str
    source_hash: str
    ground_truth_hash: str
    split: Literal["train"]
    recipe_hash: str | None = None
    severity: int | None = None
    seed: int | None = None
    transform_log_hash: str | None = None
    failure_id: str | None = None
    generation_mode: Literal["clean", "offline", "online"] = "clean"


class TrainingDatasetManifest(_FrozenContract):
    manifest_id: str
    manifest_hash: str
    dataset_version_id: str
    split_manifest_hash: str
    defense_profile_id: str
    entries: tuple[TrainingManifestEntry, ...]
    distribution_report: TrainingDatasetEstimate
    leakage_report_hash: str
    schema_version: Literal["1.0.0"] = "1.0.0"


class TrainingDatasetBuilder:
    def estimate(self, config: TrainingDatasetConfig) -> TrainingDatasetEstimate:
        validated = self._validated(config)
        generated = validated["generated"]
        clean_ids = self._select_clean_ids(config, len(generated))
        estimated_bytes = sum(validated["generated_bytes"])
        classes = Counter(
            label
            for record in config.base_version.records
            if record.sample_id in clean_ids
            for label in record.class_labels
        )
        recipes = Counter(record.recipe_hash for record, _ in generated)
        severities = Counter(
            step.severity
            for record, _ in generated
            for step in record.ordered_steps
        )
        violations = (
            ("storage_budget_exceeded",)
            if estimated_bytes > config.storage_budget_bytes
            else ()
        )
        warnings = (
            ("no_generated_variants",) if not generated else ()
        )
        return TrainingDatasetEstimate(
            clean_count=len(clean_ids),
            generated_count=len(generated),
            estimated_bytes=estimated_bytes,
            online_count=0,
            offline_count=len(generated),
            class_distribution=dict(sorted(classes.items())),
            object_size_distribution={},
            recipe_distribution=dict(sorted(recipes.items())),
            severity_distribution=dict(sorted(severities.items())),
            warnings=warnings,
            hard_cap_violations=violations,
        )

    def build(self, config: TrainingDatasetConfig) -> TrainingDatasetManifest:
        estimate = self.estimate(config)
        if estimate.hard_cap_violations:
            raise ValueError(
                f"storage hard cap violated: {estimate.hard_cap_violations}"
            )
        validated = self._validated(config)
        generated = validated["generated"]
        clean_ids = self._select_clean_ids(config, len(generated))
        records_by_id = {
            record.sample_id: record for record in config.base_version.records
        }
        entries: list[TrainingManifestEntry] = [
            TrainingManifestEntry(
                entry_id=f"clean:{sample_id}",
                kind="clean",
                source_sample_id=sample_id,
                source_dataset_version_id=config.base_version.version_id,
                artifact_path=records_by_id[sample_id].source_uri,
                source_hash=records_by_id[sample_id].source_hash,
                ground_truth_hash=records_by_id[sample_id].ground_truth_hash,
                split="train",
            )
            for sample_id in clean_ids
        ]
        for record, root in generated:
            first_step = record.ordered_steps[0]
            entries.append(
                TrainingManifestEntry(
                    entry_id=f"generated:{record.variant_id}",
                    kind="generated",
                    source_sample_id=record.source_sample_id,
                    source_dataset_version_id=record.source_dataset_version_id,
                    artifact_path=str(Path(root) / record.image_path),
                    source_hash=record.source_hash,
                    ground_truth_hash=record.ground_truth_hash,
                    split="train",
                    recipe_hash=record.recipe_hash,
                    severity=first_step.severity,
                    seed=first_step.derived_seed,
                    transform_log_hash=stable_digest(
                        record.transform_logs,
                        length=64,
                    ),
                    generation_mode="offline",
                )
            )
        ordered_entries = tuple(sorted(entries, key=lambda entry: entry.entry_id))
        leakage_payload = {
            "dataset_version_id": config.base_version.version_id,
            "split_manifest_hash": config.split_manifest.manifest_hash,
            "source_ids": sorted(entry.source_sample_id for entry in ordered_entries),
        }
        leakage_report_hash = stable_digest(leakage_payload, length=64)
        payload = {
            "dataset_version_id": config.base_version.version_id,
            "split_manifest_hash": config.split_manifest.manifest_hash,
            "defense_profile_id": config.defense_profile.profile_id,
            "entries": [entry.model_dump(mode="json") for entry in ordered_entries],
            "distribution_report": estimate.model_dump(mode="json"),
            "leakage_report_hash": leakage_report_hash,
            "schema_version": config.schema_version,
        }
        manifest_hash = stable_digest(payload, length=64)
        manifest = TrainingDatasetManifest(
            manifest_id=f"training-{manifest_hash[:20]}",
            manifest_hash=manifest_hash,
            entries=ordered_entries,
            distribution_report=estimate,
            leakage_report_hash=leakage_report_hash,
            **{key: value for key, value in payload.items() if key not in {"entries", "distribution_report", "leakage_report_hash"}},
        )
        output = Path(config.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "training-manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return manifest

    def _validated(self, config: TrainingDatasetConfig) -> dict[str, Any]:
        split_report = SplitBuilder().validate(
            config.base_version,
            config.split_manifest,
        )
        if not split_report.passed:
            raise ValueError(f"split validation failed: {split_report.errors}")
        base = {record.sample_id: record for record in config.base_version.records}
        train_ids = set(config.split_manifest.train_ids)
        locked_ids = set(config.split_manifest.locked_test_ids)
        seen_variants: set[str] = set()
        per_source: Counter[str] = Counter()
        per_recipe: Counter[str] = Counter()
        generated: list[tuple[GeneratedVariantRecord, str]] = []
        generated_bytes: list[int] = []
        for source in config.generated_sources:
            root = Path(source.root).expanduser().resolve()
            for record in source.records:
                if record.variant_id in seen_variants:
                    raise ValueError(f"duplicate generated variant: {record.variant_id}")
                seen_variants.add(record.variant_id)
                if record.source_sample_id in locked_ids:
                    raise ValueError("locked test sample cannot enter training")
                if record.source_sample_id not in train_ids:
                    raise ValueError("generated source is not in the training split")
                if record.intended_use != "training":
                    raise ValueError("benchmark/review artifact cannot enter training")
                if record.status != "complete" or record.validation_status != "passed":
                    raise ValueError("generated variant is not validated and complete")
                if not record.ordered_steps or not record.intermediate_hashes:
                    raise ValueError("generated variant is missing transform execution logs")
                if (
                    config.defense_profile.recipe_ids
                    and record.recipe_hash not in config.defense_profile.recipe_ids
                ):
                    raise ValueError("generated recipe is not allowed by defense profile")
                source_record = base.get(record.source_sample_id)
                if source_record is None:
                    raise ValueError("generated variant has a missing source")
                if source_record.source_hash != record.source_hash:
                    raise ValueError("generated source hash mismatch")
                if source_record.ground_truth_hash != record.ground_truth_hash:
                    raise ValueError("generated ground-truth hash mismatch")
                for step in record.ordered_steps:
                    entry = CATALOG_ENTRIES.get(step.attack_name)
                    if (
                        entry is None
                        or entry.expected_version != step.implementation_version
                        or entry.production_status != "production"
                    ):
                        raise ValueError("generated variant uses an incompatible attack")
                image_path = root / record.image_path
                label_path = root / record.label_path
                if not image_path.is_file() or not label_path.is_file():
                    raise ValueError("generated artifact file is missing")
                image = np.load(image_path, allow_pickle=False)
                if array_digest(image, length=32) != record.output_hash:
                    raise ValueError("generated image hash mismatch")
                label_payload = json.loads(label_path.read_text(encoding="utf-8"))
                if stable_digest(label_payload, length=32) != record.label_hash:
                    raise ValueError("generated ground-truth file hash mismatch")
                per_source[record.source_sample_id] += 1
                if per_source[record.source_sample_id] > config.defense_profile.max_variants_per_source:
                    raise ValueError("max variants per source exceeded")
                per_recipe[record.recipe_hash] += 1
                if per_recipe[record.recipe_hash] > config.defense_profile.max_variants_per_recipe:
                    raise ValueError("max variants per recipe exceeded")
                generated.append((record, source.root))
                generated_bytes.append(image_path.stat().st_size + label_path.stat().st_size)
        if sum(generated_bytes) > config.storage_budget_bytes:
            raise ValueError("storage budget exceeded")
        return {"generated": generated, "generated_bytes": generated_bytes}

    @staticmethod
    def _select_clean_ids(
        config: TrainingDatasetConfig,
        generated_count: int,
    ) -> tuple[str, ...]:
        candidates = tuple(sorted(config.split_manifest.train_ids))
        if not candidates:
            return ()
        if generated_count == 0 or config.defense_profile.generated_ratio == 0:
            return candidates
        desired = round(
            generated_count
            * config.defense_profile.clean_replay_ratio
            / config.defense_profile.generated_ratio
        )
        desired = max(1, min(len(candidates), desired))
        return candidates[:desired]
