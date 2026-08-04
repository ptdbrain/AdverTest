"""Group E: person-vanishing physical-plausible patch application."""

from __future__ import annotations

from typing import ClassVar

from src.adapters.base import ModelAdapter
from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.attacks.patch._utils import (
    apply_eot_transform,
    load_patch,
    place_patch,
    sample_eot_transform,
    select_box,
)
from src.core.types import AttackGroup, CostClass, Sample


class ThysPatchParams(AttackParams):
    patch_path: str | None = None
    artifact_hash: str | None = None
    area_fraction_per_severity: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)
    target_label: str = "Pedestrian"
    eot: bool = True
    allow_builtin_patch: bool = False


@ATTACKS.register
class ThysPatch(BaseAttack):
    """Apply a person-vanishing patch with deterministic EOT transforms."""

    name: ClassVar[str] = "thys_patch"
    version: ClassVar[str] = "2.0.0"
    group: ClassVar[AttackGroup] = "E"
    cost_class: ClassVar[CostClass] = "cheap"
    severity_levels: ClassVar[int] = 4
    required_annotations = frozenset({"boxes"})
    generation_mode: ClassVar[str] = "artifact"
    owner: ClassVar[str] = "group-d-e"
    reference: ClassVar[str] = "Thys et al., CVPRW 2019 (arXiv:1904.08653)"
    params_model: ClassVar[type[AttackParams]] = ThysPatchParams

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.patch, self.artifact_hash = load_patch(
            self.params.patch_path,
            expected_hash=self.params.artifact_hash,
            required_algorithm=None if self.params.allow_builtin_patch else self.name,
            required_objective=(
                None if self.params.allow_builtin_patch else "vanishing"
            ),
            required_source_label=(
                None if self.params.allow_builtin_patch else "Pedestrian"
            ),
            required_target_label=(
                None
                if self.params.allow_builtin_patch
                else self.params.target_label
            ),
            allow_builtin=self.params.allow_builtin_patch,
        )

    def validate_requirements(
        self,
        sample: Sample,
        model: ModelAdapter | None,
    ) -> None:
        super().validate_requirements(sample, model)
        select_box(sample, self.params.target_label)

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        box = select_box(sample, self.params.target_label)
        area_fraction = self.level(
            severity,
            self.params.area_fraction_per_severity,
        )
        patch = self.patch
        mask = None
        if self.params.eot:
            transformed = apply_eot_transform(
                self.patch,
                sample_eot_transform(ctx.rng),
            )
            patch = transformed.image
            mask = transformed.mask
            area_fraction *= transformed.transform.scale**2
        attacked, _ = place_patch(
            sample,
            patch,
            box,
            area_fraction=min(1.0, area_fraction),
            rng=ctx.rng,
            random_offset=self.params.eot,
            mask=mask,
        )
        return attacked
