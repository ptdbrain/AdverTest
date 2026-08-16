"""Ordered, deterministic execution of ordinary versioned attack recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.adapters.base import ModelAdapter
from src.attacks import ATTACK_CATALOG, get_attack, load_attacks
from src.attacks.base import AttackContext
from src.attacks.catalog import AttackCatalog
from src.attacks.recipes import AttackRecipe, AttackRecipeStep, RecipeBuilder
from src.core.hashing import array_digest, stable_digest
from src.core.objectives import AttackObjective
from src.core.types import Modality, Sample, Task, validate_image
from src.pipeline.annotations import (
    AnnotationPolicy,
    AnnotationTransformer,
    AnnotationTransformLog,
)


@dataclass(frozen=True, slots=True)
class CompositionContext:
    run_seed: int
    model: ModelAdapter | None = None
    objective: AttackObjective = AttackObjective()
    annotation_policy: AnnotationPolicy = AnnotationPolicy()
    fail_fast: bool = True
    max_cost: float = float("inf")
    online: bool = False
    available_artifacts: frozenset[str] | None = None


class CompositionStepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    position: int
    attack_name: str
    implementation_version: str
    severity: int
    probability: float
    requested_seed: int
    derived_seed: int
    applied: bool
    status: str
    resolved_parameters: dict[str, Any] = Field(default_factory=dict)
    cost: float = Field(ge=0.0)
    input_hash: str
    output_hash: str | None = None
    transform_log: AnnotationTransformLog | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True, eq=False)
class CompositionResult:
    final_sample: Sample | None
    step_records: tuple[CompositionStepRecord, ...]
    intermediate_arrays: tuple[np.ndarray, ...]
    intermediate_hashes: tuple[str, ...]
    transform_logs: tuple[AnnotationTransformLog, ...]
    resolved_parameters: tuple[dict[str, Any], ...]
    total_cost: float
    errors: tuple[str, ...]
    loadable: bool


class CompositionExecutionError(RuntimeError):
    def __init__(self, result: CompositionResult) -> None:
        self.result = result
        detail = result.errors[0] if result.errors else "composition failed"
        super().__init__(detail)


class CompositionEngine:
    def __init__(
        self,
        *,
        catalog: AttackCatalog | None = None,
        annotation_transformer: AnnotationTransformer | None = None,
    ) -> None:
        load_attacks()
        self.catalog = catalog or ATTACK_CATALOG
        self.annotation_transformer = (
            annotation_transformer or AnnotationTransformer()
        )

    def execute(
        self,
        sample: Sample,
        recipe: AttackRecipe,
        context: CompositionContext,
    ) -> CompositionResult:
        task, capabilities, annotations, modality = _compatibility_context(
            sample,
            context.model,
        )
        preflight = RecipeBuilder().validate(
            recipe,
            self.catalog,
            task=task,
            model_capabilities=capabilities,
            annotation_types=annotations,
            modality=modality,
            online=context.online,
            available_artifacts=context.available_artifacts,
        )
        if not preflight.valid:
            result = _preflight_failure(sample, recipe, context, preflight.errors)
            if context.fail_fast:
                raise CompositionExecutionError(result)
            return result

        filtered = self.catalog.list(
            task=task,
            model_capabilities=capabilities,
            annotation_types=annotations,
            modality=modality,
            online=context.online,
            available_artifacts=context.available_artifacts,
        )
        exclusions = {item.name: item.reasons for item in filtered.exclusions}
        current = sample
        records: list[CompositionStepRecord] = []
        arrays: list[np.ndarray] = []
        hashes: list[str] = []
        transform_logs: list[AnnotationTransformLog] = []
        resolved_parameters: list[dict[str, Any]] = []
        errors: list[str] = []
        total_cost = 0.0
        cumulative_occlusion = 0.0

        for index, step in enumerate(recipe.steps):
            input_hash = array_digest(current.image)
            derived_seed = _derived_seed(sample, recipe, step, context.run_seed)
            try:
                metadata = self.catalog.get(step.attack_name)
                if metadata.implementation_version != step.implementation_version:
                    raise ValueError(
                        f"implementation version mismatch for {step.attack_name!r}"
                    )
                reasons = exclusions.get(step.attack_name, ())
                if reasons:
                    raise ValueError(
                        f"attack {step.attack_name!r} is incompatible: {', '.join(reasons)}"
                    )
                attack_params = {
                    key: value
                    for key, value in step.parameters.items()
                    if key not in {"spatial_transform", "occlusion_ratio"}
                }
                attack = get_attack(step.attack_name, **attack_params)
                resolved = attack.resolve_parameters(step.severity)
                resolved.update(
                    {
                        key: value
                        for key, value in step.parameters.items()
                        if key in {"spatial_transform", "occlusion_ratio"}
                    }
                )
                resolved_parameters.append(resolved)
                rng = np.random.default_rng(derived_seed)
                selected = step.probability >= 1.0 or (
                    step.probability > 0.0 and float(rng.random()) < step.probability
                )
                if not selected:
                    output_hash = array_digest(current.image)
                    records.append(
                        _step_record(
                            step,
                            derived_seed,
                            applied=False,
                            status="not_selected_by_probability",
                            resolved=resolved,
                            cost=0.0,
                            input_hash=input_hash,
                            output_hash=output_hash,
                        )
                    )
                    _append_intermediate(current.image, arrays, hashes)
                    continue

                if total_cost + step.expected_cost > context.max_cost:
                    raise ValueError(
                        f"composition cost would exceed {context.max_cost}"
                    )
                next_occlusion = cumulative_occlusion + float(
                    step.parameters.get("occlusion_ratio", 0.0)
                )
                if next_occlusion > recipe.constraints.max_occlusion_ratio:
                    raise ValueError("cumulative occlusion ratio exceeded")
                attack_context = AttackContext(
                    rng=rng,
                    model=context.model,
                    objective=step.objective or context.objective,
                )
                spatial_transform = attack.spatial_transform(
                    current,
                    step.severity,
                    attack_context,
                )
                attacked = attack.run(current, step.severity, attack_context)
                _validate_attack_ground_truth(current, attacked, attack.name)
                transform_log = None
                if spatial_transform is not None:
                    attacked, transform_log = self.annotation_transformer.apply(
                        attacked,
                        spatial_transform,
                        context.annotation_policy,
                    )
                    transform_logs.append(transform_log)
                validate_image(attacked.image)
                total_cost += step.expected_cost
                cumulative_occlusion = next_occlusion
                current = attacked
                output_hash = array_digest(current.image)
                records.append(
                    _step_record(
                        step,
                        derived_seed,
                        applied=True,
                        status="completed",
                        resolved=resolved,
                        cost=step.expected_cost,
                        input_hash=input_hash,
                        output_hash=output_hash,
                        transform_log=transform_log,
                    )
                )
                _append_intermediate(current.image, arrays, hashes)
            except Exception as exc:
                message = (
                    f"step {step.position} {step.attack_name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                errors.append(message)
                records.append(
                    _step_record(
                        step,
                        derived_seed,
                        applied=False,
                        status="failed",
                        resolved=(
                            resolved_parameters[-1]
                            if len(resolved_parameters) > step.position
                            else {}
                        ),
                        cost=0.0,
                        input_hash=input_hash,
                        error=message,
                    )
                )
                for blocked in recipe.steps[index + 1 :]:
                    records.append(
                        _step_record(
                            blocked,
                            _derived_seed(
                                sample,
                                recipe,
                                blocked,
                                context.run_seed,
                            ),
                            applied=False,
                            status="blocked_by_prior_failure",
                            resolved={},
                            cost=0.0,
                            input_hash=input_hash,
                            error="blocked by prior failed step",
                        )
                    )
                result = CompositionResult(
                    final_sample=None,
                    step_records=tuple(records),
                    intermediate_arrays=tuple(arrays),
                    intermediate_hashes=tuple(hashes),
                    transform_logs=tuple(transform_logs),
                    resolved_parameters=tuple(resolved_parameters),
                    total_cost=total_cost,
                    errors=tuple(errors),
                    loadable=False,
                )
                if context.fail_fast:
                    raise CompositionExecutionError(result) from exc
                return result

        return CompositionResult(
            final_sample=current,
            step_records=tuple(records),
            intermediate_arrays=tuple(arrays),
            intermediate_hashes=tuple(hashes),
            transform_logs=tuple(transform_logs),
            resolved_parameters=tuple(resolved_parameters),
            total_cost=total_cost,
            errors=(),
            loadable=True,
        )


def _compatibility_context(
    sample: Sample,
    model: ModelAdapter | None,
) -> tuple[
    Task,
    frozenset[Any],
    frozenset[Any],
    Modality,
]:
    task: Task = (
        model.metadata().task
        if model is not None
        else "segmentation"
        if sample.mask is not None
        else "detection2d"
    )
    capabilities = model.capabilities if model is not None else frozenset()
    annotations = frozenset(
        annotation
        for annotation, available in (
            ("boxes", bool(sample.boxes)),
            ("mask", sample.mask is not None),
        )
        if available
    )
    has_lidar = sample.lidar_frame is not None or sample.lidar is not None
    modality: Modality = (
        "multi"
        if has_lidar and (sample.camera_views or sample.image is not None)
        else "lidar"
        if has_lidar
        else "image"
    )
    return task, capabilities, annotations, modality


def _derived_seed(
    sample: Sample,
    recipe: AttackRecipe,
    step: AttackRecipeStep,
    run_seed: int,
) -> int:
    return int(
        stable_digest(
            {
                "run_seed": run_seed,
                "step_seed": step.seed,
                "sample_id": sample.sample_id,
                "recipe_hash": recipe.recipe_hash,
                "catalog_version": recipe.catalog_version,
                "position": step.position,
                "attack": step.attack_name,
                "implementation_version": step.implementation_version,
            },
            length=16,
        ),
        16,
    ) % (2**32)


def _step_record(
    step: AttackRecipeStep,
    derived_seed: int,
    *,
    applied: bool,
    status: str,
    resolved: dict[str, Any],
    cost: float,
    input_hash: str,
    output_hash: str | None = None,
    transform_log: AnnotationTransformLog | None = None,
    error: str | None = None,
) -> CompositionStepRecord:
    return CompositionStepRecord(
        position=step.position,
        attack_name=step.attack_name,
        implementation_version=step.implementation_version,
        severity=step.severity,
        probability=step.probability,
        requested_seed=step.seed,
        derived_seed=derived_seed,
        applied=applied,
        status=status,
        resolved_parameters=resolved,
        cost=cost,
        input_hash=input_hash,
        output_hash=output_hash,
        transform_log=transform_log,
        error=error,
    )


def _append_intermediate(
    image: np.ndarray,
    arrays: list[np.ndarray],
    hashes: list[str],
) -> None:
    evidence = np.ascontiguousarray(image).copy()
    evidence.setflags(write=False)
    arrays.append(evidence)
    hashes.append(array_digest(evidence))


def _validate_attack_ground_truth(
    source: Sample,
    attacked: Sample,
    attack_name: str,
) -> None:
    if attacked.boxes != source.boxes or attacked.boxes3d != source.boxes3d:
        raise ValueError(f"attack {attack_name!r} changed ground-truth boxes")
    if source.mask is None or attacked.mask is None:
        same_mask = source.mask is attacked.mask
    else:
        same_mask = bool(np.array_equal(source.mask, attacked.mask))
    if not same_mask:
        raise ValueError(f"attack {attack_name!r} changed the ground-truth mask")


def _preflight_failure(
    sample: Sample,
    recipe: AttackRecipe,
    context: CompositionContext,
    errors: tuple[str, ...],
) -> CompositionResult:
    input_hash = array_digest(sample.image)
    records = tuple(
        _step_record(
            step,
            _derived_seed(sample, recipe, step, context.run_seed),
            applied=False,
            status="validation_failed",
            resolved={},
            cost=0.0,
            input_hash=input_hash,
            error="; ".join(errors),
        )
        for step in recipe.steps
    )
    return CompositionResult(
        final_sample=None,
        step_records=records,
        intermediate_arrays=(),
        intermediate_hashes=(),
        transform_logs=(),
        resolved_parameters=(),
        total_cost=0.0,
        errors=errors,
        loadable=False,
    )
