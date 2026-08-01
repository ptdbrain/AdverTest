"""Test-run orchestration: dataset x attack x severity -> model -> metrics.

The runner is the only place that knows about all four plugin kinds, which is
what keeps contributors independent: an attack author never edits this file.

Flow (plan §4): resolve plugins -> anonymisation gate -> cost estimate ->
clean predictions (cached) -> per-cell variants -> AP -> :class:`RunReport`.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.adapters import get_adapter
from src.adapters.base import ModelAdapter
from src.attacks import get_attack, load_attacks
from src.attacks.base import AttackContext, BaseAttack
from src.core.hashing import clean_key, stable_digest, variant_key
from src.core.types import COST_WEIGHT, ModelInfo, Prediction, Sample
from src.datasets import get_dataset
from src.datasets.base import DatasetSource
from src.evaluation.detection_metrics import DEFAULT_IOU_THRESHOLD, average_precision
from src.evaluation.report import CellResult, RunReport, SkippedAttack
from src.pipeline.cache import MemoryCache, PredictionCache


class RunConfig(BaseModel):
    """One test run. Empty ``attacks`` means "every compatible attack"."""

    model_config = ConfigDict(extra="forbid")

    model: str = "blob_detector"
    adapter_params: dict[str, Any] = Field(default_factory=dict)
    dataset: str = "synthetic_shapes"
    dataset_params: dict[str, Any] = Field(default_factory=dict)
    attacks: list[str] = Field(default_factory=list)
    attack_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    severities: list[int] = Field(default_factory=lambda: [1, 3, 5])
    limit: int | None = 8
    seed: int = 20260730
    iou_threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, gt=0.0, lt=1.0)


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Shown before a run starts — plan §5 forbids "estimate afterwards"."""

    n_cells: int
    n_samples: int
    n_forward_passes: int
    cost_units: float
    estimated_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_samples": self.n_samples,
            "n_forward_passes": self.n_forward_passes,
            "cost_units": round(self.cost_units, 2),
            "estimated_seconds": round(self.estimated_seconds, 2),
        }


class TestRunner:
    """Executes a :class:`RunConfig` and returns a :class:`RunReport`."""

    def __init__(self, cache: PredictionCache | None = None, *, seconds_per_cost_unit: float = 0.05) -> None:
        self.cache = cache if cache is not None else MemoryCache()
        self.seconds_per_cost_unit = seconds_per_cost_unit

    # ------------------------------------------------------------- estimation

    def estimate(self, config: RunConfig) -> CostEstimate:
        """Cost of the run without executing anything."""
        dataset = get_dataset(config.dataset, **config.dataset_params)
        adapter = get_adapter(config.model, **config.adapter_params)
        n_samples = len(dataset.load(config.limit))
        selected, _ = self._resolve_attacks(config, dataset, adapter.metadata())
        n_cells = len(selected) * len(config.severities)
        units = sum(
            COST_WEIGHT[attack.cost_class] * n_samples * len(config.severities) for attack in selected
        )
        return CostEstimate(
            n_cells=n_cells,
            n_samples=n_samples,
            n_forward_passes=n_samples * (n_cells + 1),
            cost_units=units + n_samples,
            estimated_seconds=(units + n_samples) * self.seconds_per_cost_unit,
        )

    # ------------------------------------------------------------- execution

    def run(self, config: RunConfig) -> RunReport:
        started = perf_counter()
        dataset = get_dataset(config.dataset, **config.dataset_params)
        dataset.require_anonymized()
        adapter = get_adapter(config.model, **config.adapter_params)
        info = adapter.metadata()
        samples = dataset.load(config.limit)
        if not samples:
            raise ValueError(f"dataset {config.dataset!r} returned no samples")

        clean = self._predict_clean(adapter, samples, info)
        report = RunReport(
            run_id=uuid.uuid4().hex[:12],
            model=info.name,
            model_version=info.version,
            dataset=dataset.name,
            n_samples=len(samples),
            ap_clean=average_precision(clean, samples, config.iou_threshold),
        )
        selected, skipped = self._resolve_attacks(config, dataset, info)
        report.skipped = skipped
        for attack_cls in selected:
            attack = get_attack(attack_cls.name, **config.attack_params.get(attack_cls.name, {}))
            report.cells.extend(self._run_attack(attack, samples, adapter, config))
        report.seconds = perf_counter() - started
        return report

    def _run_attack(
        self,
        attack: BaseAttack,
        samples: Sequence[Sample],
        adapter: ModelAdapter,
        config: RunConfig,
    ) -> list[CellResult]:
        """One row of the heatmap: the same attack at every requested severity."""
        cells: list[CellResult] = []
        for severity in config.severities:
            started = perf_counter()
            hits_before = self.cache.hits
            variants = [self._attack_sample(attack, sample, severity, config, adapter) for sample in samples]
            predictions = self._predict_variants(adapter, variants, attack, severity)
            cells.append(
                CellResult(
                    attack=attack.name,
                    group=attack.group,
                    severity=severity,
                    ap=average_precision(predictions, samples, config.iou_threshold),
                    n_samples=len(samples),
                    seconds=perf_counter() - started,
                    cache_hits=self.cache.hits - hits_before,
                )
            )
        return cells

    def _attack_sample(
        self,
        attack: BaseAttack,
        sample: Sample,
        severity: int,
        config: RunConfig,
        adapter: ModelAdapter,
    ) -> Sample:
        """Deterministic per-(run, sample, attack, severity) randomness."""
        seed_material = stable_digest(
            {
                "seed": config.seed,
                "sample": sample.sample_id,
                "attack": attack.name,
                "severity": severity,
            }
        )
        rng = np.random.default_rng(int(seed_material, 16) % (2**32))
        context = AttackContext(rng=rng, model=adapter if attack.needs_model else None)
        return attack.run(sample, severity, context)

    # -------------------------------------------------------------- inference

    def _predict_clean(
        self,
        adapter: ModelAdapter,
        samples: Sequence[Sample],
        info: ModelInfo,
    ) -> list[Prediction]:
        keys = [clean_key(sample_id=sample.sample_id, model_version=info.version) for sample in samples]
        return self._predict_cached(adapter, samples, keys)

    def _predict_variants(
        self,
        adapter: ModelAdapter,
        variants: Sequence[Sample],
        attack: BaseAttack,
        severity: int,
    ) -> list[Prediction]:
        version = adapter.metadata().version
        keys = [
            variant_key(
                sample_id=variant.sample_id,
                attack=attack.name,
                params=attack.param_dict(),
                severity=severity,
                model_version=version,
            )
            for variant in variants
        ]
        return self._predict_cached(adapter, variants, keys)

    def _predict_cached(
        self,
        adapter: ModelAdapter,
        samples: Sequence[Sample],
        keys: Sequence[str],
    ) -> list[Prediction]:
        """Batch only the cache misses, then restore the original order."""
        results: dict[int, Prediction] = {}
        pending: list[tuple[int, Sample, str]] = []
        for index, (sample, key) in enumerate(zip(samples, keys, strict=True)):
            cached = self.cache.get(key)
            if cached is None:
                pending.append((index, sample, key))
            else:
                results[index] = cached
        if pending:
            fresh = adapter.predict([sample for _, sample, _ in pending])
            for (index, _, key), prediction in zip(pending, fresh, strict=True):
                self.cache.put(key, prediction)
                results[index] = prediction
        return [results[index] for index in range(len(samples))]

    # ------------------------------------------------------------- resolution

    @staticmethod
    def _resolve_attacks(
        config: RunConfig,
        dataset: DatasetSource,
        info: ModelInfo,
    ) -> tuple[list[type[BaseAttack]], list[SkippedAttack]]:
        """Pick the attacks to run, recording why each other one was skipped."""
        catalog = load_attacks()
        requested = [catalog.get(name) for name in config.attacks] if config.attacks else catalog.values()
        selected: list[type[BaseAttack]] = []
        skipped: list[SkippedAttack] = []
        for attack in requested:
            reason = _incompatibility(attack, dataset, info)
            params = config.attack_params.get(attack.name, {})
            if (
                reason is None
                and attack.generation_mode == "artifact"
                and not params.get("patch_path")
                and not params.get("allow_builtin_patch", False)
            ):
                reason = "artifact attack requires patch_path"
            if reason is None:
                selected.append(attack)
            else:
                skipped.append(SkippedAttack(attack.name, reason))
        return selected, skipped


def _incompatibility(attack: type[BaseAttack], dataset: DatasetSource, info: ModelInfo) -> str | None:
    """Reason this attack cannot run here, or ``None`` when it can."""
    if attack.needs_gradients and not info.supports_gradients:
        return f"attack needs input gradients, adapter {info.name!r} does not expose them"
    if attack.modality != "image" and attack.modality != dataset.modality:
        return f"attack modality {attack.modality!r} != dataset modality {dataset.modality!r}"
    return None
