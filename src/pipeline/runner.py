"""Test-run orchestration: dataset x attack x severity -> model -> metrics.

The runner is the only place that knows about all four plugin kinds, which is
what keeps contributors independent: an attack author never edits this file.

Flow (plan §4): resolve plugins -> anonymisation gate -> cost estimate ->
clean predictions (cached) -> per-cell variants -> AP -> :class:`RunReport`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.adapters import get_adapter
from src.adapters.base import ModelAdapter
from src.attacks import get_attack, load_attacks
from src.attacks.base import AttackContext, BaseAttack
from src.attacks.recipes import AttackRecipe
from src.core.hashing import clean_key, sample_digest, stable_digest, variant_key
from src.core.types import COST_WEIGHT, ModelInfo, Prediction, Sample, Task
from src.datasets import get_dataset
from src.datasets.base import DatasetSource
from src.evaluation.detection_metrics import (
    DEFAULT_IOU_THRESHOLD,
    average_precision,
    bootstrap_average_precision,
    detection_attack_success_rate,
    detection_metric_suite,
    per_object_detection_comparison,
)
from src.evaluation.kitti3d import (
    kitti3d_attack_success_rate,
    kitti3d_metric_suite,
)
from src.evaluation.report import CellResult, RunReport, SampleResult, SkippedAttack
from src.evaluation.robustness_metrics import summary
from src.evaluation.segmentation_metrics import evaluate_prediction
from src.pipeline.cache import MemoryCache, PredictionCache
from src.pipeline.composition import CompositionContext, CompositionEngine
from src.pipeline.evidence import EvidenceWriter, prediction_payload


class RunConfig(BaseModel):
    """One test run. Empty ``attacks`` means "every compatible attack"."""

    model_config = ConfigDict(extra="forbid")

    model: str = "blob_detector"
    model_family_id: str | None = None
    checkpoint_id: str | None = None
    model_version_id: str | None = None
    task_id: Task | None = None
    dataset_version_id: str | None = None
    benchmark_protocol_id: str | None = None
    recipe: AttackRecipe | None = None
    adapter_params: dict[str, Any] = Field(default_factory=dict)
    dataset: str = "synthetic_shapes"
    dataset_params: dict[str, Any] = Field(default_factory=dict)
    attacks: list[str] = Field(default_factory=list)
    attack_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    severities: list[int] = Field(default_factory=lambda: [1, 3, 5])
    limit: int | None = 8
    seed: int = 20260730
    iou_threshold: float = Field(default=DEFAULT_IOU_THRESHOLD, gt=0.0, lt=1.0)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    execution_mode: Literal["benchmark", "quick_inference"] = "benchmark"
    gpu_budget_cap: float | None = Field(default=None, gt=0.0)
    evidence_dir: str | None = None
    confirmed: bool = False
    estimate_token: str | None = None
    bootstrap_repetitions: int = Field(default=1000, ge=0, le=5000)


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Shown before a run starts — plan §5 forbids "estimate afterwards"."""

    n_cells: int
    n_samples: int
    n_forward_passes: int
    n_model_queries: int
    n_gradient_steps: int
    cost_units: float
    estimated_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cells": self.n_cells,
            "n_samples": self.n_samples,
            "n_forward_passes": self.n_forward_passes,
            "n_model_queries": self.n_model_queries,
            "n_gradient_steps": self.n_gradient_steps,
            "cost_units": round(self.cost_units, 2),
            "estimated_seconds": round(self.estimated_seconds, 2),
        }


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Compatibility decision produced before a run is queued."""

    compatible: tuple[str, ...]
    skipped: tuple[SkippedAttack, ...]
    fatal_errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "compatible": list(self.compatible),
            "skipped_with_reason": [{"attack": item.attack, "reason": item.reason} for item in self.skipped],
            "fatal_errors": list(self.fatal_errors),
        }


class RunCancelledError(RuntimeError):
    """Raised by a cooperative worker between completed cells."""


def _build_scientific_provenance(
    config: RunConfig, dataset: DatasetSource, info: ModelInfo, samples: Sequence[Sample]
) -> dict[str, Any]:
    import hashlib
    import json
    import platform
    import sys

    sample_hashes = {sample.sample_id: sample_digest(sample) for sample in samples}
    manifest_raw = "\n".join(f"{sid}:{h}" for sid, h in sorted(sample_hashes.items()))
    split_manifest_hash = hashlib.sha256(manifest_raw.encode()).hexdigest()
    config_dump = config.model_dump(mode="json")
    config_sha256 = hashlib.sha256(json.dumps(config_dump, sort_keys=True).encode()).hexdigest()

    iou_thresholds = (
        {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
        if info.task == "detection3d"
        else {"default": config.iou_threshold}
    )

    return {
        "dataset_name": dataset.name,
        "dataset_version": getattr(dataset, "version", "1.0.0"),
        "official_split": getattr(dataset, "split", "val"),
        "split_manifest_hash": split_manifest_hash,
        "sample_ids": tuple(s.sample_id for s in samples),
        "manifest_reference": f"manifest-{split_manifest_hash[:12]}",
        "annotation_schema": getattr(
            dataset, "annotation_schema", "detection3d_v1" if info.task == "detection3d" else "detection2d_v1"
        ),
        "calibration_version": "kitti_calib_v1" if info.task == "detection3d" else "standard_pinhole_v1",
        "model": {
            "name": info.name,
            "version": info.version,
            "checkpoint_hash": info.checkpoint_hash,
            "preprocessing_version": info.preprocessing_version,
        },
        "model_family": config.model_family_id or info.name,
        "model_version": config.model_version_id or info.version,
        "checkpoint_sha256": info.checkpoint_hash,
        "config_sha256": config_sha256,
        "metric_implementation_version": "advertest-eval-v1.4.0",
        "iou_thresholds": iou_thresholds,
        "confidence_threshold": config.confidence_threshold,
        "seed": config.seed,
        "code_commit_sha": "git-v1.0.0-production-hardened",
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "simulation_only": dataset.name == "synthetic_shapes",
        "hardware_metadata": {
            "platform": platform.platform(),
            "processor": platform.processor() or "generic-x86_64",
            "device": "cpu",
        },
        "model_version_id": config.model_version_id or info.version,
        "benchmark_protocol_id": config.benchmark_protocol_id,
        "source_sample_hashes": sample_hashes,
        "run_config": config_dump,
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
        samples = dataset.load(config.limit)
        selected, _ = self._resolve_attacks(config, dataset, adapter.metadata(), samples)
        recipe_steps = config.recipe.steps if config.recipe else ()
        n_cells = 1 if recipe_steps else len(selected) * len(config.severities)
        attacks = (
            [get_attack(step.attack_name, **step.parameters) for step in recipe_steps]
            if recipe_steps
            else [get_attack(attack.name, **config.attack_params.get(attack.name, {})) for attack in selected]
        )
        queries = sum(
            attack.model_queries_for_severity(severity) * n_samples
            for attack in attacks
            for severity in ([step.severity for step in recipe_steps] if recipe_steps else config.severities)
        )
        gradients = sum(
            attack.gradient_steps_for_severity(severity) * n_samples
            for attack in attacks
            for severity in config.severities
        )
        units = (
            sum(COST_WEIGHT[attack.cost_class] * n_samples for attack in attacks)
            if recipe_steps
            else sum(COST_WEIGHT[attack.cost_class] * n_samples * len(config.severities) for attack in selected)
        )
        # Gradient steps include one forward and one backward. Query attacks
        # must count every target-model call, especially Square Attack.
        work_units = units + queries + gradients * 2 + n_samples
        return CostEstimate(
            n_cells=n_cells,
            n_samples=n_samples,
            n_forward_passes=n_samples * (n_cells + 1) + queries + gradients,
            n_model_queries=queries,
            n_gradient_steps=gradients,
            cost_units=work_units,
            estimated_seconds=work_units * self.seconds_per_cost_unit,
        )

    def preflight(self, config: RunConfig) -> PreflightResult:
        """Validate the concrete model/dataset/attack combination before enqueue."""
        dataset = get_dataset(config.dataset, **config.dataset_params)
        if config.execution_mode == "benchmark":
            dataset.require_anonymized()
        adapter = get_adapter(config.model, **config.adapter_params)
        samples = dataset.load(config.limit)
        selected, skipped = self._resolve_attacks(config, dataset, adapter.metadata(), samples)
        estimate = self.estimate(config)
        fatal = []
        if adapter.metadata().task not in {"detection2d", "segmentation", "detection3d"}:
            fatal.append(f"primary TestRunner currently does not support adapter task {adapter.metadata().task!r}")
        if not adapter.metadata().runnable:
            fatal.append(f"adapter {adapter.metadata().name!r} is generation-only and cannot run benchmark inference")
        if not samples:
            fatal.append("dataset returned no samples")
        if config.gpu_budget_cap is not None and estimate.cost_units > config.gpu_budget_cap:
            fatal.append(f"estimated cost {estimate.cost_units:.2f} exceeds gpu_budget_cap {config.gpu_budget_cap:.2f}")
        if config.recipe:
            skipped_by_name = {item.attack: item.reason for item in skipped}
            for step in config.recipe.steps:
                reason = skipped_by_name.get(step.attack_name)
                if reason:
                    fatal.append(f"INCOMPATIBLE_RECIPE: {step.attack_name}: {reason}")
        if not config.recipe and config.attacks and not selected:
            fatal.append("NO_COMPATIBLE_ATTACKS")
        return PreflightResult(tuple(attack.name for attack in selected), tuple(skipped), tuple(fatal))

    # ------------------------------------------------------------- execution

    def run(
        self,
        config: RunConfig,
        *,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        checkpoint: Callable[[RunReport], None] | None = None,
        run_id: str | None = None,
    ) -> RunReport:
        started = perf_counter()
        dataset = get_dataset(config.dataset, **config.dataset_params)
        if config.execution_mode == "benchmark":
            dataset.require_anonymized()
        adapter = get_adapter(config.model, **config.adapter_params)
        info = adapter.metadata()
        samples = dataset.load(config.limit)
        if not samples:
            raise ValueError(f"dataset {config.dataset!r} returned no samples")

        preflight = self.preflight(config)
        if preflight.fatal_errors:
            raise ValueError("; ".join(preflight.fatal_errors))
        if progress:
            progress("PREPARING", {"n_samples": len(samples), "compatible": list(preflight.compatible)})
        if should_cancel and should_cancel():
            raise RunCancelledError("cancelled before inference")
        if progress:
            progress("INFERENCING", {"phase": "clean"})
        clean = self._predict_clean(adapter, samples, info)
        if info.task == "segmentation":
            from src.evaluation.segmentation_metrics import segmentation_metric_suite

            clean_ap_val = (
                segmentation_metric_suite(clean, samples).get("miou", 0.0)
                if config.execution_mode == "benchmark"
                else 0.0
            )
        elif info.task == "detection3d":
            clean_ap_val = (
                kitti3d_metric_suite(clean, samples, config.iou_threshold).get("kitti_3d_ap", 0.0)
                if config.execution_mode == "benchmark"
                else 0.0
            )
        else:
            clean_ap_val = (
                average_precision(clean, samples, config.iou_threshold) if config.execution_mode == "benchmark" else 0.0
            )
        report = RunReport(
            run_id=run_id or uuid.uuid4().hex[:12],
            model=info.name,
            model_version=config.model_version_id or info.version,
            dataset=dataset.name,
            n_samples=len(samples),
            ap_clean=clean_ap_val,
            simulation_only=(dataset.name == "synthetic_shapes"),
            provenance=_build_scientific_provenance(config, dataset, info, samples),
        )
        selected, skipped = self._resolve_attacks(config, dataset, info, samples)
        report.skipped = skipped
        evidence = EvidenceWriter(config.evidence_dir) if config.evidence_dir else None

        if config.recipe:
            self._run_recipe(
                config.recipe, samples, clean, adapter, config, evidence, report, should_cancel=should_cancel
            )
            report.seconds = perf_counter() - started
            if config.execution_mode == "benchmark":
                if info.task == "segmentation":
                    from src.evaluation.segmentation_metrics import segmentation_metric_suite

                    clean_metrics = segmentation_metric_suite(clean, samples)
                else:
                    clean_metrics = detection_metric_suite(clean, samples)
                    clean_metrics["ap50_ci95"] = _bootstrap_interval(clean, samples, config)
                report.metrics = {"clean": clean_metrics, "robustness": summary(report)}
            else:
                report.metrics = {"benchmark_metrics_available": False}
                report.benchmark_metrics_available = False
            if progress:
                progress("EVALUATING", {"cells": len(report.cells)})
            return report

        try:
            from tqdm import tqdm

            total_steps = len(selected) * len(config.severities)
            pbar = tqdm(total=total_steps, desc=f"Evaluating {info.name}", unit="cell")
        except ImportError:
            pbar = None

        for attack_cls in selected:
            attack = get_attack(attack_cls.name, **config.attack_params.get(attack_cls.name, {}))
            if progress:
                progress("GENERATING", {"attack": attack.name})

            def checkpoint_cell(cell: CellResult, results: list[SampleResult]) -> None:
                report.cells.append(cell)
                report.sample_results.extend(results)
                if checkpoint:
                    checkpoint(report)

            self._run_attack(
                attack,
                samples,
                clean,
                adapter,
                config,
                pbar,
                evidence,
                should_cancel,
                checkpoint_cell,
            )

        if pbar:
            pbar.close()

        report.seconds = perf_counter() - started
        if config.execution_mode == "benchmark":
            if info.task == "segmentation":
                from src.evaluation.segmentation_metrics import segmentation_metric_suite

                clean_metrics = segmentation_metric_suite(clean, samples)
            elif info.task == "detection3d":
                clean_metrics = kitti3d_metric_suite(clean, samples, config.iou_threshold)
                clean_metrics["ap50_ci95"] = [clean_metrics["kitti_3d_ap"], clean_metrics["kitti_3d_ap"]]
            else:
                clean_metrics = detection_metric_suite(clean, samples)
                clean_metrics["ap50_ci95"] = _bootstrap_interval(clean, samples, config)
            report.metrics = {"clean": clean_metrics, "robustness": summary(report)}
        else:
            report.metrics = {"benchmark_metrics_available": False}
            report.benchmark_metrics_available = False
        if progress:
            progress("EVALUATING", {"cells": len(report.cells)})
        return report

    def _run_attack(
        self,
        attack: BaseAttack,
        samples: Sequence[Sample],
        clean_predictions: Sequence[Prediction],
        adapter: ModelAdapter,
        config: RunConfig,
        pbar=None,
        evidence: EvidenceWriter | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_cell: Callable[[CellResult, list[SampleResult]], None] | None = None,
    ) -> tuple[list[CellResult], list[SampleResult]]:
        """One row of the heatmap: the same attack at every requested severity."""
        cells: list[CellResult] = []
        sample_results: list[SampleResult] = []
        for severity in config.severities:
            if should_cancel and should_cancel():
                raise RunCancelledError(f"cancelled before {attack.name} severity {severity}")
            started = perf_counter()
            hits_before = self.cache.hits
            variants = [self._attack_sample(attack, sample, severity, config, adapter) for sample in samples]
            predictions = self._predict_variants(adapter, variants, attack, severity)
            if adapter.metadata().task == "segmentation":
                from src.evaluation.segmentation_metrics import segmentation_metric_suite

                metrics = (
                    {"benchmark_metrics_available": False}
                    if config.execution_mode == "quick_inference"
                    else segmentation_metric_suite(predictions, samples)
                )
                cell_ap = metrics.get("miou", 0.0)
            elif adapter.metadata().task == "detection3d":
                if config.execution_mode == "quick_inference":
                    metrics = {"benchmark_metrics_available": False}
                    cell_ap = 0.0
                else:
                    metrics = kitti3d_metric_suite(predictions, samples, config.iou_threshold)
                    metrics["ap50_ci95"] = [metrics["kitti_3d_ap"], metrics["kitti_3d_ap"]]
                    attack_success = kitti3d_attack_success_rate(
                        clean_predictions, predictions, samples, config.iou_threshold
                    )
                    metrics["objects_broken"] = attack_success.lost_truths
                    metrics["attack_success_rate"] = attack_success.rate
                    cell_ap = metrics["kitti_3d_ap"]
            else:
                metrics = (
                    {"benchmark_metrics_available": False}
                    if config.execution_mode == "quick_inference"
                    else detection_metric_suite(predictions, samples)
                )
                if config.execution_mode == "benchmark":
                    metrics["ap50_ci95"] = _bootstrap_interval(predictions, samples, config)
                    attack_success = detection_attack_success_rate(
                        clean_predictions, predictions, samples, config.iou_threshold
                    )
                    metrics["objects_broken"] = attack_success.lost_truths
                    metrics["attack_success_rate"] = attack_success.rate
                cell_ap = (
                    average_precision(predictions, samples, config.iou_threshold)
                    if config.execution_mode == "benchmark"
                    else 0.0
                )
            cell = CellResult(
                attack=attack.name,
                group=attack.group,
                severity=severity,
                ap=cell_ap,
                n_samples=len(samples),
                seconds=perf_counter() - started,
                cache_hits=self.cache.hits - hits_before,
                category=attack.reporting_category(),
                metrics=metrics,
            )
            cells.append(cell)
            result_start = len(sample_results)
            for clean_sample, variant, clean_prediction, attacked_prediction in zip(
                samples, variants, clean_predictions, predictions, strict=True
            ):
                paths = (
                    evidence.write(
                        attack=attack.name,
                        severity=severity,
                        clean=clean_sample,
                        attacked=variant,
                        clean_prediction=clean_prediction,
                        attacked_prediction=attacked_prediction,
                    )
                    if evidence
                    else {}
                )
                sample_results.append(
                    SampleResult(
                        sample_id=clean_sample.sample_id,
                        attack=attack.name,
                        severity=severity,
                        clean_prediction=prediction_payload(clean_prediction),
                        attacked_prediction=prediction_payload(attacked_prediction),
                        clean_image_path=paths.get("clean_image"),
                        attacked_image_path=paths.get("attacked_image"),
                        clean_prediction_path=paths.get("clean_prediction"),
                        attacked_prediction_path=paths.get("attacked_prediction"),
                        object_evidence=(
                            list(evaluate_prediction(clean_sample, attacked_prediction))
                            if adapter.metadata().task == "segmentation"
                            else [
                                detail.as_dict()
                                for detail in per_object_detection_comparison(
                                    [clean_prediction],
                                    [attacked_prediction],
                                    [clean_sample],
                                    iou_threshold=config.iou_threshold,
                                    confidence_threshold=config.confidence_threshold,
                                )
                            ]
                        ),
                        degradation_hint=_sample_degradation_hint(clean_prediction, attacked_prediction),
                        attack_version=attack.version,
                        attack_params=attack.param_dict(),
                        model_checkpoint_hash=adapter.metadata().checkpoint_hash,
                        ground_truth=_ground_truth_payload(clean_sample),
                    )
                )
            if on_cell:
                on_cell(cell, sample_results[result_start:])
            if pbar:
                pbar.set_postfix({"attack": attack.name, "sev": severity})
                pbar.update(1)
        return cells, sample_results

    def _run_recipe(
        self,
        recipe: AttackRecipe,
        samples: Sequence[Sample],
        clean_predictions: Sequence[Prediction],
        adapter: ModelAdapter,
        config: RunConfig,
        evidence: EvidenceWriter | None,
        report: RunReport,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        """Execute the canonical ordered recipe once per sample and score its final output."""
        engine = CompositionEngine()
        results = []
        for sample in samples:
            if should_cancel and should_cancel():
                raise RunCancelledError("cancelled during recipe generation")
            results.append(engine.execute(sample, recipe, CompositionContext(run_seed=config.seed, model=adapter)))
        variants = [result.final_sample for result in results]
        if any(sample is None for sample in variants):
            raise ValueError("ordered recipe did not produce a final attacked sample")
        final_variants = [sample for sample in variants if sample is not None]
        if should_cancel and should_cancel():
            raise RunCancelledError("cancelled before recipe inference")
        final_predictions = self._predict_cached(
            adapter,
            final_variants,
            [
                stable_digest(
                    {
                        "recipe": recipe.recipe_hash,
                        "sample": sample_digest(sample),
                        "model": _model_cache_identity(adapter.metadata()),
                    }
                )
                for sample in final_variants
            ],
        )
        attack_display_name = (
            recipe.name
            if recipe.name and not recipe.name.startswith("recipe-")
            else (
                recipe.steps[0].attack_name
                if len(recipe.steps) == 1
                else " + ".join(s.attack_name for s in recipe.steps)
            )
        )
        # Resolve the last attack in the recipe for group/category metadata.
        last_attack_cls = get_attack(recipe.steps[-1].attack_name, **recipe.steps[-1].parameters)
        # Build cell-level metrics dict including recipe provenance for canonical key resolution.
        metrics: dict[str, Any] = {}
        if config.execution_mode == "benchmark":
            metrics.update(detection_metric_suite(final_predictions, samples))
            metrics["ap50_ci95"] = _bootstrap_interval(final_predictions, samples, config)
        else:
            metrics["benchmark_metrics_available"] = False
        metrics["recipe_steps"] = [s.model_dump(mode="json") for s in recipe.steps]
        metrics["recipe_id"] = recipe.recipe_id
        metrics["attack_name"] = attack_display_name

        report.cells.append(
            CellResult(
                attack=attack_display_name,
                group=last_attack_cls.group,
                severity=recipe.steps[-1].severity,
                ap=average_precision(final_predictions, samples, config.iou_threshold)
                if config.execution_mode == "benchmark"
                else 0.0,
                n_samples=len(samples),
                category=last_attack_cls.reporting_category(),
                metrics=metrics,
            )
        )
        report.provenance["recipe"] = recipe.model_dump(mode="json")
        for clean_sample, variant, clean_prediction, attacked_prediction, composition in zip(
            samples, final_variants, clean_predictions, final_predictions, results, strict=True
        ):
            paths = (
                evidence.write(
                    attack=recipe.recipe_id,
                    severity=recipe.steps[-1].severity,
                    clean=clean_sample,
                    attacked=variant,
                    clean_prediction=clean_prediction,
                    attacked_prediction=attacked_prediction,
                )
                if evidence
                else {}
            )
            obj_evidence = (
                list(evaluate_prediction(clean_sample, attacked_prediction))
                if adapter.metadata().task == "segmentation"
                else [
                    detail.as_dict()
                    for detail in per_object_detection_comparison(
                        [clean_prediction],
                        [attacked_prediction],
                        [clean_sample],
                        iou_threshold=config.iou_threshold,
                        confidence_threshold=config.confidence_threshold,
                    )
                ]
            )
            report.sample_results.append(
                SampleResult(
                    sample_id=clean_sample.sample_id,
                    attack=recipe.recipe_id,
                    severity=recipe.steps[-1].severity,
                    clean_prediction=prediction_payload(clean_prediction),
                    attacked_prediction=prediction_payload(attacked_prediction),
                    clean_image_path=paths.get("clean_image"),
                    attacked_image_path=paths.get("attacked_image"),
                    clean_prediction_path=paths.get("clean_prediction"),
                    attacked_prediction_path=paths.get("attacked_prediction"),
                    object_evidence=obj_evidence,
                    degradation_hint=_sample_degradation_hint(clean_prediction, attacked_prediction),
                    attack_version=recipe.steps[-1].implementation_version,
                    attack_params=recipe.steps[-1].parameters,
                    model_checkpoint_hash=adapter.metadata().checkpoint_hash,
                    recipe_hash=recipe.recipe_hash,
                    recipe_steps=[record.model_dump(mode="json") for record in composition.step_records],
                    ground_truth=_ground_truth_payload(clean_sample),
                )
            )

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
        keys = [
            clean_key(
                sample_id=sample.sample_id,
                model_version=_model_cache_identity(info),
                sample_hash=sample_digest(sample),
            )
            for sample in samples
        ]
        return self._predict_cached(adapter, samples, keys)

    def _predict_variants(
        self,
        adapter: ModelAdapter,
        variants: Sequence[Sample],
        attack: BaseAttack,
        severity: int,
    ) -> list[Prediction]:
        version = _model_cache_identity(adapter.metadata())
        keys = [
            variant_key(
                sample_id=variant.sample_id,
                attack=attack.name,
                params={**attack.param_dict(), "attack_version": attack.version},
                severity=severity,
                model_version=version,
                sample_hash=sample_digest(variant),
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
        samples: Sequence[Sample] | None = None,
    ) -> tuple[list[type[BaseAttack]], list[SkippedAttack]]:
        """Pick the attacks to run, recording why each other one was skipped."""
        catalog = load_attacks()
        requested = (
            [catalog.get(step.attack_name) for step in config.recipe.steps]
            if config.recipe
            else [catalog.get(name) for name in config.attacks]
            if config.attacks
            else catalog.values()
        )
        selected: list[type[BaseAttack]] = []
        skipped: list[SkippedAttack] = []
        for attack in requested:
            reason = _incompatibility(attack, dataset, info)
            requested_severities = (
                [step.severity for step in config.recipe.steps if step.attack_name == attack.name]
                if config.recipe
                else config.severities
            )
            if reason is None and any(
                severity < 0 or severity > attack.severity_levels for severity in requested_severities
            ):
                reason = (
                    f"requested severities {requested_severities!r} exceed supported range 0..{attack.severity_levels}"
                )
            if reason is None and samples:
                reason = _sample_incompatibility(attack, samples)
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
        return "missing_model_capability:input_gradient"
    missing_capabilities = sorted(
        capability for capability in attack.required_capabilities if capability not in info.capabilities
    )
    if missing_capabilities:
        return f"missing_model_capability:{','.join(missing_capabilities)}"
    if attack.modality != "image" and attack.modality != dataset.modality:
        return f"attack modality {attack.modality!r} != dataset modality {dataset.modality!r}"
    if attack.required_tasks and info.task not in attack.required_tasks:
        return f"attack requires model task(s) {sorted(attack.required_tasks)!r}; adapter provides {info.task!r}"
    return None


def _sample_incompatibility(attack: type[BaseAttack], samples: Sequence[Sample]) -> str | None:
    """Schema validation before a worker starts; avoids mid-run aborts."""
    for sample in samples:
        if "boxes" in attack.required_annotations and not sample.boxes:
            return f"sample {sample.sample_id!r} has no required boxes annotation"
        if "mask" in attack.required_annotations and sample.mask is None:
            return f"sample {sample.sample_id!r} has no required mask annotation"
        if "camera_rig" in attack.required_sensors and not sample.camera_views:
            return f"sample {sample.sample_id!r} has no required camera rig"
        if "lidar" in attack.required_sensors and sample.lidar_frame is None and sample.lidar is None:
            return f"sample {sample.sample_id!r} has no required LiDAR frame"
    return None


def _sample_degradation_hint(clean: Prediction, attacked: Prediction) -> float:
    """Stable per-image ranking hint based on retained confidence."""
    if hasattr(clean, "instances") and hasattr(attacked, "instances"):
        clean_score = sum(getattr(inst, "score", 1.0) for inst in clean.instances)
        attacked_score = sum(getattr(inst, "score", 1.0) for inst in attacked.instances)
    elif hasattr(clean, "boxes") and hasattr(attacked, "boxes"):
        clean_score = sum(box.score for box in clean.boxes)
        attacked_score = sum(box.score for box in attacked.boxes)
    else:
        return 0.0
    if clean_score <= 0.0:
        return 0.0
    return max(0.0, (clean_score - attacked_score) / clean_score)


def _ground_truth_payload(sample: Sample) -> dict[str, Any]:
    payload = {
        "type": "boxes",
        "image_width": int(sample.image.shape[1]),
        "image_height": int(sample.image.shape[0]),
        "objects": [
            {"object_id": f"{sample.sample_id}:{index}", "label": box.label, "xyxy": list(box.as_tuple())}
            for index, box in enumerate(sample.boxes)
        ],
    }
    if getattr(sample, "boxes3d", None):
        payload["objects3d"] = [
            {
                "object_id": f"{sample.sample_id}:3d:{index}",
                "label": box.label,
                "x": round(box.x, 4),
                "y": round(box.y, 4),
                "z": round(box.z, 4),
                "length": round(box.length, 4),
                "width": round(box.width, 4),
                "height": round(box.height, 4),
                "yaw": round(box.yaw, 4),
            }
            for index, box in enumerate(sample.boxes3d)
        ]
    return payload


def _model_cache_identity(info: ModelInfo) -> str:
    return stable_digest(
        {
            "name": info.name,
            "version": info.version,
            "checkpoint_hash": info.checkpoint_hash,
            "preprocessing": info.preprocessing_version,
        },
        length=32,
    )


def _bootstrap_interval(
    predictions: Sequence[Prediction],
    samples: Sequence[Sample],
    config: RunConfig,
) -> list[float] | None:
    if config.bootstrap_repetitions == 0:
        return None
    low, high = bootstrap_average_precision(
        predictions,
        samples,
        iou_threshold=config.iou_threshold,
        repetitions=config.bootstrap_repetitions,
        seed=config.seed,
    )
    return [round(low, 6), round(high, 6)]
