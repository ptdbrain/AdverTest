"""Attack plugin contract (plan §2).

Every corruption, weather effect, occlusion, adversarial perturbation or patch
is one subclass of :class:`BaseAttack` living in **its own file**. The pipeline
only ever calls :meth:`BaseAttack.run`, which enforces the shared invariants so
a bug in one plugin cannot corrupt a whole test run:

* ``severity == 0`` is always a no-op (sanity check #1 of plan §3),
* ``1 <= severity <= severity_levels`` is validated,
* white-box attacks get a model or a clear error,
* the returned image is clipped to ``[0, 1]``, cast to float32, and checked for
  shape/dtype/NaN.

Subclasses implement :meth:`apply` only. See ``src/attacks/_template.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict

from src.adapters.base import ModelAdapter
from src.core.objectives import (
    AttackObjective,
    RequiredAnnotation,
    SurrogateCapability,
)
from src.core.types import (
    MAX_SEVERITY,
    AttackGroup,
    CostClass,
    LidarFrame,
    Modality,
    Sample,
    SensorKind,
    Task,
    validate_image,
)


class AttackParams(BaseModel):
    """Base class for per-attack parameter models.

    ``extra="forbid"`` turns a typo in a config file into an immediate error
    instead of a silently ignored setting; ``frozen=True`` keeps a configured
    attack instance safe to reuse across samples and threads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelRequiredError(RuntimeError):
    """A white-box attack was run without a model adapter in the context."""


@dataclass(frozen=True, slots=True)
class AttackContext:
    """Everything an attack may use besides the sample itself.

    ``rng`` is seeded per (run, sample, attack, severity) by the runner, so
    results are reproducible and attacks must never use global ``np.random``.
    """

    rng: np.random.Generator
    model: ModelAdapter | None = None
    objective: AttackObjective = AttackObjective()

    def require_model(self, attack_name: str) -> ModelAdapter:
        if self.model is None:
            raise ModelRequiredError(f"attack {attack_name!r} needs a model adapter in the context")
        return self.model


class BaseAttack(ABC):
    """One attack / corruption plugin."""

    #: Registry key: snake_case, unique across the whole catalog.
    name: ClassVar[str]
    version: ClassVar[str] = "1.0.0"
    #: Plan §2 group: A corruptions, B weather, C occlusion, D white-box, E patch, F black-box.
    group: ClassVar[AttackGroup]
    modality: ClassVar[Modality] = "image"
    #: Inputs and payloads touched by the attack.  ``modality`` remains the
    #: compatibility/catalog summary used by older clients.
    required_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"image"})
    affected_sensors: ClassVar[frozenset[SensorKind]] = frozenset({"image"})
    cost_class: ClassVar[CostClass] = "cheap"
    severity_levels: ClassVar[int] = MAX_SEVERITY
    #: True when :meth:`apply` calls ``ctx.model`` (white-box / query-based).
    needs_model: ClassVar[bool] = False
    #: True when gradients are required, so black-box adapters are skipped.
    needs_gradients: ClassVar[bool] = False
    required_annotations: ClassVar[frozenset[RequiredAnnotation]] = frozenset()
    required_capabilities: ClassVar[frozenset[SurrogateCapability]] = frozenset()
    #: Model tasks accepted by model-dependent attacks. Empty means any task.
    required_tasks: ClassVar[frozenset[Task]] = frozenset()
    generation_mode: ClassVar[str] = "per_sample"
    #: Team member who owns this file — the only "assignment table" we need.
    owner: ClassVar[str] = "unassigned"
    #: Paper / library the implementation follows.
    reference: ClassVar[str] = ""
    params_model: ClassVar[type[AttackParams]] = AttackParams

    def __init__(self, **params: Any) -> None:
        self.params = self.params_model(**params)

    # ------------------------------------------------------------- public API

    def run(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        """Apply the attack and enforce the shared contract. Do not override."""
        if severity == 0:
            return sample
        if not 1 <= severity <= self.severity_levels:
            raise ValueError(
                f"severity for {self.name!r} must be 0..{self.severity_levels}, got {severity}"
            )
        self.validate_requirements(sample, ctx.model)
        attacked = self.apply(sample, severity, ctx)
        image = np.clip(attacked.image, 0.0, 1.0).astype(np.float32, copy=False)
        validate_image(image, like=sample.image)
        # Image attacks retain the historical image contract.  Sensor-only
        # attacks may intentionally leave the canonical image unchanged.
        if "image" in self.affected_sensors:
            attacked = attacked.with_image(image)
        elif not np.array_equal(attacked.image, sample.image):
            raise ValueError(f"attack {self.name!r} changed an undeclared image payload")
        if "lidar" in self.affected_sensors:
            if attacked.lidar_frame is None:
                raise ValueError(f"attack {self.name!r} did not return a LiDAR frame")
            validate_lidar_frame(attacked.lidar_frame)
        return attacked

    def validate_requirements(
        self,
        sample: Sample,
        model: ModelAdapter | None,
    ) -> None:
        """Fail before generation when annotations or model capabilities are absent."""
        if self.needs_model:
            if model is None:
                raise ModelRequiredError(
                    f"attack {self.name!r} needs a model adapter in the context"
                )
            missing = sorted(
                capability
                for capability in self.required_capabilities
                if not model.supports(capability)
            )
            if missing:
                raise ModelRequiredError(
                    f"attack {self.name!r} requires surrogate capabilities: "
                    f"{', '.join(missing)}"
                )
            if self.required_tasks and model.metadata().task not in self.required_tasks:
                required = ", ".join(sorted(self.required_tasks))
                raise ModelRequiredError(
                    f"attack {self.name!r} requires model task(s): {required}; "
                    f"got {model.metadata().task!r}"
                )
        missing_annotations = [
            annotation
            for annotation in self.required_annotations
            if (annotation == "boxes" and not sample.boxes)
            or (annotation == "mask" and sample.mask is None)
        ]
        if missing_annotations:
            raise ValueError(
                f"attack {self.name!r} requires annotations: "
                f"{', '.join(sorted(missing_annotations))}"
            )
        if "camera_rig" in self.required_sensors and not sample.camera_views:
            raise ValueError(f"attack {self.name!r} requires camera views")
        if "lidar" in self.required_sensors and sample.lidar_frame is None and sample.lidar is None:
            raise ValueError(f"attack {self.name!r} requires a LiDAR frame")

    @abstractmethod
    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        """Return a new sample with attacked pixels. Never mutate ``sample``."""

    # -------------------------------------------------------------- helpers

    def level(self, severity: int, values: Sequence[float]) -> float:
        """Pick the per-severity value, tolerating short parameter tuples."""
        if not values:
            raise ValueError(f"{self.name}: empty per-severity value list")
        return float(values[min(severity, len(values)) - 1])

    def param_dict(self) -> dict[str, Any]:
        """Configured parameters, used in the cache key."""
        return self.params.model_dump(mode="json")

    def model_queries_for_severity(self, severity: int) -> int:
        """Worst-case inference queries for one sample at ``severity``.

        Most attacks either have no model dependency or use gradients, which are
        accounted for separately. Query-based attacks override this so generation
        estimates and manifests state the exact black-box budget.
        """
        return 0

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Catalog entry for the API / CLI (plan §2 plugin declaration)."""
        return {
            "name": cls.name,
            "version": cls.version,
            "group": cls.group,
            "title": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
            "modality": cls.modality,
            "required_sensors": sorted(cls.required_sensors),
            "affected_sensors": sorted(cls.affected_sensors),
            "cost_class": cls.cost_class,
            "severity_levels": cls.severity_levels,
            "needs_model": cls.needs_model,
            "needs_gradients": cls.needs_gradients,
            "required_annotations": sorted(cls.required_annotations),
            "required_capabilities": sorted(cls.required_capabilities),
            "required_tasks": sorted(cls.required_tasks),
            "generation_mode": cls.generation_mode,
            "owner": cls.owner,
            "reference": cls.reference,
            "params_schema": cls.params_model.model_json_schema(),
        }


def validate_lidar_frame(frame: LidarFrame) -> None:
    """Validate the multimodal point-cloud contract."""
    if not isinstance(frame.points, np.ndarray):
        raise TypeError("LiDAR points must be a numpy array")
    if frame.points.ndim != 2 or frame.points.shape[1] != len(frame.fields):
        raise ValueError("LiDAR points must be shaped (N, len(fields))")
    if not np.issubdtype(frame.points.dtype, np.floating):
        raise ValueError("LiDAR points must use a floating dtype")
    if not np.isfinite(frame.points).all():
        raise ValueError("LiDAR points contain NaN or inf")
