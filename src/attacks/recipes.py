"""Immutable attack recipes, validation, estimates, and seeded builders."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.attacks.catalog import AttackCatalog, AttackCatalogError, AttackMetadata
from src.core.hashing import stable_digest
from src.core.objectives import AttackObjective, RequiredAnnotation, SurrogateCapability
from src.core.types import AttackGroup, Modality, Task


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class RecipeConstraints(_FrozenContract):
    max_steps: int = Field(default=4, ge=1)
    max_white_box_steps: int = Field(default=1, ge=0)
    max_expensive_steps: int = Field(default=1, ge=0)
    max_occlusion_ratio: float = Field(default=0.60, ge=0.0, le=1.0)
    max_variants: int = Field(default=10_000, ge=1)
    max_storage_bytes: int = Field(default=100_000_000_000, ge=1)
    max_gpu_hours: float = Field(default=1_000.0, ge=0.0)
    supported_spatial_transforms: tuple[str, ...] = (
        "identity",
        "crop",
        "translate",
        "scale",
        "horizontal_flip",
        "rotate_90",
        "affine",
        "resize_pad",
    )


class AttackRecipeStep(_FrozenContract):
    position: int = Field(ge=0)
    attack_name: str
    implementation_version: str
    severity: int = Field(ge=0, le=5)
    parameters: dict[str, Any] = Field(default_factory=dict)
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int = Field(ge=0)
    objective: AttackObjective | None = None
    expected_cost: float = Field(ge=0.0)


class AttackRecipe(_FrozenContract):
    name: str
    catalog_version: str = "1.0.0"
    steps: tuple[AttackRecipeStep, ...]
    constraints: RecipeConstraints = Field(default_factory=RecipeConstraints)
    recipe_id: str = ""
    recipe_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def derive_identity(self) -> Self:
        positions = tuple(step.position for step in self.steps)
        if positions != tuple(range(len(self.steps))):
            raise ValueError("recipe step positions must be unique and contiguous from zero")
        payload = {
            "name": self.name,
            "catalog_version": self.catalog_version,
            "steps": [step.model_dump(mode="json") for step in self.steps],
            "constraints": self.constraints.model_dump(mode="json"),
            "metadata": self.metadata,
        }
        recipe_hash = stable_digest(payload, length=64)
        recipe_id = f"recipe-{recipe_hash[:20]}"
        if self.recipe_hash and self.recipe_hash != recipe_hash:
            raise ValueError("recipe_hash does not match canonical recipe content")
        if self.recipe_id and self.recipe_id != recipe_id:
            raise ValueError("recipe_id does not match canonical recipe content")
        object.__setattr__(self, "recipe_hash", recipe_hash)
        object.__setattr__(self, "recipe_id", recipe_id)
        return self


class RecipeEstimate(_FrozenContract):
    variant_count: int = Field(ge=0)
    estimated_bytes: int = Field(ge=0)
    estimated_gpu_hours: float = Field(ge=0.0)
    expected_cost_units: float = Field(ge=0.0)


class RecipeValidation(_FrozenContract):
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    estimate: RecipeEstimate


class RandomNRequest(_FrozenContract):
    count: int = Field(ge=1, le=1_000)
    steps_per_recipe: int = Field(default=2, ge=1)
    seed: int = Field(ge=0)
    allowlist: tuple[str, ...] = ()
    blocklist: tuple[str, ...] = ()
    required_attacks: tuple[str, ...] = ()
    severity_min: int = Field(default=1, ge=0, le=5)
    severity_max: int = Field(default=5, ge=0, le=5)
    task: Task = "detection2d"
    model_capabilities: frozenset[SurrogateCapability] = frozenset()
    annotation_types: frozenset[RequiredAnnotation] = frozenset()
    modality: Modality = "multi"
    online: bool = False
    constraints: RecipeConstraints = Field(default_factory=RecipeConstraints)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.severity_min > self.severity_max:
            raise ValueError("severity_min must not exceed severity_max")
        if self.steps_per_recipe > self.constraints.max_steps:
            raise ValueError("steps_per_recipe exceeds recipe max_steps")
        if self.count > self.constraints.max_variants:
            raise ValueError("requested recipe count exceeds variant hard cap")
        return self


class StratifiedRandomRequest(_FrozenContract):
    count: int = Field(ge=1, le=1_000)
    group_quotas: dict[AttackGroup, int]
    seed: int = Field(ge=0)
    allowlist: tuple[str, ...] = ()
    blocklist: tuple[str, ...] = ()
    severity_min: int = Field(default=1, ge=0, le=5)
    severity_max: int = Field(default=5, ge=0, le=5)
    task: Task = "detection2d"
    model_capabilities: frozenset[SurrogateCapability] = frozenset()
    annotation_types: frozenset[RequiredAnnotation] = frozenset()
    modality: Modality = "multi"
    online: bool = False
    constraints: RecipeConstraints = Field(default_factory=RecipeConstraints)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.group_quotas or any(quota <= 0 for quota in self.group_quotas.values()):
            raise ValueError("group quotas must be positive")
        if sum(self.group_quotas.values()) > self.constraints.max_steps:
            raise ValueError("group quotas exceed recipe max_steps")
        if self.severity_min > self.severity_max:
            raise ValueError("severity_min must not exceed severity_max")
        return self


class SweepRequest(_FrozenContract):
    attack_names: tuple[str, ...]
    severities: tuple[int, ...]
    seed: int = Field(ge=0)
    constraints: RecipeConstraints = Field(default_factory=RecipeConstraints)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if not self.attack_names or len(set(self.attack_names)) != len(self.attack_names):
            raise ValueError("sweep attack_names must be non-empty and unique")
        if not self.severities or any(not 0 <= severity <= 5 for severity in self.severities):
            raise ValueError("sweep severities must be non-empty and in 0..5")
        if len(self.attack_names) * len(self.severities) > self.constraints.max_variants:
            raise ValueError("sweep exceeds variant hard cap")
        return self


class RecipeBuilder:
    def validate(
        self,
        recipe: AttackRecipe,
        catalog: AttackCatalog,
        *,
        task: Task,
        model_capabilities: frozenset[SurrogateCapability],
        annotation_types: frozenset[RequiredAnnotation],
        modality: Modality,
        online: bool,
        requested_variants: int = 1,
        bytes_per_variant: int = 0,
        available_artifacts: frozenset[str] | None = None,
    ) -> RecipeValidation:
        errors: list[str] = []
        warnings: list[str] = []
        names = [step.attack_name for step in recipe.steps]
        for name, count in sorted(Counter(names).items()):
            if count > 1:
                errors.append(f"duplicate_attack:{name}")
        if len(recipe.steps) > recipe.constraints.max_steps:
            errors.append("recipe_length_exceeded")

        filtered = catalog.list(
            task=task,
            model_capabilities=model_capabilities,
            annotation_types=annotation_types,
            modality=modality,
            online=online,
            available_artifacts=available_artifacts,
        )
        exclusions = {item.name: item.reasons for item in filtered.exclusions}
        metadata: list[AttackMetadata] = []
        metadata_by_name: dict[str, AttackMetadata] = {}
        for step in recipe.steps:
            try:
                item = catalog.get(step.attack_name)
            except AttackCatalogError:
                errors.append(f"unknown_attack:{step.attack_name}")
                continue
            metadata.append(item)
            metadata_by_name[step.attack_name] = item
            if item.implementation_version != step.implementation_version:
                errors.append(f"implementation_version_mismatch:{step.attack_name}")
            errors.extend(
                f"incompatible:{step.attack_name}:{reason}"
                for reason in exclusions.get(step.attack_name, ())
            )
            transform = step.parameters.get("spatial_transform")
            if (
                transform is not None
                and transform not in recipe.constraints.supported_spatial_transforms
            ):
                errors.append(f"unsupported_spatial_transform:{transform}")

        white_box_count = sum(item.group == "D" for item in metadata)
        expensive_count = sum(item.cost_class == "expensive" for item in metadata)
        if white_box_count > recipe.constraints.max_white_box_steps:
            errors.append("multiple_white_box_steps")
        if expensive_count > recipe.constraints.max_expensive_steps:
            errors.append("multiple_expensive_steps")
        for left, right in (("fgsm", "pgd"), ("pgd", "cw_l2")):
            if {left, right}.issubset(names):
                errors.append(f"forbidden_pair:{left}+{right}")

        occlusion_ratio = sum(
            float(step.parameters.get("occlusion_ratio", 0.0))
            for step in recipe.steps
            if (item := metadata_by_name.get(step.attack_name)) is not None
            and item.group == "C"
        )
        if occlusion_ratio > recipe.constraints.max_occlusion_ratio:
            errors.append("occlusion_ratio_exceeded")

        estimate = self.estimate(
            recipe,
            requested_variants=requested_variants,
            bytes_per_variant=bytes_per_variant,
        )
        if estimate.variant_count > recipe.constraints.max_variants:
            errors.append("variant_cap_exceeded")
        if estimate.estimated_bytes > recipe.constraints.max_storage_bytes:
            errors.append("storage_cap_exceeded")
        if estimate.estimated_gpu_hours > recipe.constraints.max_gpu_hours:
            errors.append("gpu_cap_exceeded")
        if expensive_count == 1:
            warnings.append("expensive_step_requires_offline_capacity")
        if len({item.group for item in metadata} & {"A", "B", "C"}) > 1:
            warnings.append("mixed_realism_requires_visual_review")
        ordered_errors = tuple(dict.fromkeys(errors))
        ordered_warnings = tuple(dict.fromkeys(warnings))
        return RecipeValidation(
            valid=not ordered_errors,
            errors=ordered_errors,
            warnings=ordered_warnings,
            estimate=estimate,
        )

    def estimate(
        self,
        recipe: AttackRecipe,
        *,
        requested_variants: int,
        bytes_per_variant: int,
    ) -> RecipeEstimate:
        cost = sum(
            step.expected_cost * step.probability for step in recipe.steps
        ) * requested_variants
        gpu_cost = sum(
            step.expected_cost * step.probability
            for step in recipe.steps
            if step.expected_cost >= 4.0
        )
        return RecipeEstimate(
            variant_count=requested_variants,
            estimated_bytes=requested_variants * bytes_per_variant,
            estimated_gpu_hours=gpu_cost * requested_variants / 3_600.0,
            expected_cost_units=cost,
        )

    def random_n(
        self,
        request: RandomNRequest,
        catalog: AttackCatalog,
    ) -> list[AttackRecipe]:
        candidates = _candidate_metadata(request, catalog)
        required = tuple(request.required_attacks)
        by_name = {item.name: item for item in candidates}
        unavailable = sorted(set(required) - set(by_name))
        if unavailable:
            raise ValueError(f"required attacks are unavailable: {unavailable}")
        if request.steps_per_recipe < len(required):
            raise ValueError("steps_per_recipe is smaller than required_attacks")
        remaining = [item for item in candidates if item.name not in required]
        needed = request.steps_per_recipe - len(required)
        if needed > len(remaining):
            raise ValueError("not enough compatible attacks for no-replacement sampling")
        rng = random.Random(request.seed)
        recipes: list[AttackRecipe] = []
        hashes: set[str] = set()
        attempts = 0
        while len(recipes) < request.count and attempts < request.count * 100:
            attempts += 1
            chosen = [by_name[name] for name in required]
            chosen.extend(rng.sample(remaining, needed))
            rng.shuffle(chosen)
            recipe = _recipe_from_metadata(
                f"random-{request.seed}-{attempts}",
                chosen,
                rng,
                request.severity_min,
                request.severity_max,
                request.constraints,
            )
            validation = self.validate(
                recipe,
                catalog,
                task=request.task,
                model_capabilities=request.model_capabilities,
                annotation_types=request.annotation_types,
                modality=request.modality,
                online=request.online,
            )
            if validation.valid and recipe.recipe_hash not in hashes:
                hashes.add(recipe.recipe_hash)
                recipes.append(recipe)
        if len(recipes) != request.count:
            raise ValueError("could not build the requested number of valid unique recipes")
        return recipes

    def random_by_group(
        self,
        request: StratifiedRandomRequest,
        catalog: AttackCatalog,
    ) -> list[AttackRecipe]:
        candidates = _candidate_metadata(request, catalog)
        grouped: dict[AttackGroup, list[AttackMetadata]] = {
            group: [item for item in candidates if item.group == group]
            for group in request.group_quotas
        }
        for group, quota in request.group_quotas.items():
            if len(grouped[group]) < quota:
                raise ValueError(f"group {group} cannot satisfy quota {quota}")
        rng = random.Random(request.seed)
        recipes: list[AttackRecipe] = []
        hashes: set[str] = set()
        for index in range(request.count):
            chosen = [
                item
                for group, quota in sorted(request.group_quotas.items())
                for item in rng.sample(grouped[group], quota)
            ]
            rng.shuffle(chosen)
            recipe = _recipe_from_metadata(
                f"stratified-{request.seed}-{index}",
                chosen,
                rng,
                request.severity_min,
                request.severity_max,
                request.constraints,
            )
            validation = self.validate(
                recipe,
                catalog,
                task=request.task,
                model_capabilities=request.model_capabilities,
                annotation_types=request.annotation_types,
                modality=request.modality,
                online=request.online,
            )
            if not validation.valid:
                raise ValueError(
                    f"stratified recipe violates constraints: {validation.errors}"
                )
            if recipe.recipe_hash in hashes:
                raise ValueError("stratified builder produced a duplicate recipe")
            hashes.add(recipe.recipe_hash)
            recipes.append(recipe)
        return recipes

    def sweep(
        self,
        request: SweepRequest,
        catalog: AttackCatalog,
    ) -> list[AttackRecipe]:
        rng = random.Random(request.seed)
        recipes: list[AttackRecipe] = []
        for attack_name in request.attack_names:
            metadata = catalog.get(attack_name)
            for severity in request.severities:
                recipes.append(
                    AttackRecipe(
                        name=f"sweep-{attack_name}-s{severity}",
                        steps=(
                            _step_from_metadata(
                                metadata,
                                position=0,
                                severity=severity,
                                seed=rng.randrange(0, 2**63),
                            ),
                        ),
                        constraints=request.constraints,
                    )
                )
        return recipes


def _candidate_metadata(request: Any, catalog: AttackCatalog) -> list[AttackMetadata]:
    result = catalog.list(
        task=request.task,
        model_capabilities=request.model_capabilities,
        annotation_types=request.annotation_types,
        modality=request.modality,
        online=request.online,
    )
    allowlist = set(request.allowlist)
    blocklist = set(request.blocklist)
    return [
        item
        for item in result.selected
        if (not allowlist or item.name in allowlist) and item.name not in blocklist
    ]


def _recipe_from_metadata(
    name: str,
    metadata: list[AttackMetadata],
    rng: random.Random,
    severity_min: int,
    severity_max: int,
    constraints: RecipeConstraints,
) -> AttackRecipe:
    return AttackRecipe(
        name=name,
        steps=tuple(
            _step_from_metadata(
                item,
                position=position,
                severity=rng.randint(severity_min, severity_max),
                seed=rng.randrange(0, 2**63),
            )
            for position, item in enumerate(metadata)
        ),
        constraints=constraints,
    )


def _step_from_metadata(
    metadata: AttackMetadata,
    *,
    position: int,
    severity: int,
    seed: int,
) -> AttackRecipeStep:
    cost = {"cheap": 1.0, "medium": 4.0, "expensive": 20.0}[metadata.cost_class]
    return AttackRecipeStep(
        position=position,
        attack_name=metadata.name,
        implementation_version=metadata.implementation_version,
        severity=severity,
        seed=seed,
        expected_cost=cost,
    )
