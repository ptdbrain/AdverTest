"""Framework-neutral attack objectives and surrogate capabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

ObjectiveKind = Literal[
    "untargeted",
    "targeted",
    "vanishing",
    "fabrication",
    "mislabeling",
    "segmentation_bce",
    "dag",
    "cw_margin",
]
SurrogateCapability = Literal[
    "input_gradient",
    "detection_loss",
    "objectness",
    "class_logits",
    "dense_proposals",
    "segmentation_loss",
    "class_margin",
]
RequiredAnnotation = Literal["boxes", "mask"]


@dataclass(frozen=True, slots=True)
class AttackObjective:
    """What an adversarial generator asks the surrogate to maximise."""

    kind: ObjectiveKind = "untargeted"
    target_label: str | None = None
    target_box_index: int | None = None
    objective_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind in {"targeted", "mislabeling"} and not self.target_label:
            raise ValueError(f"{self.kind} objective requires target_label")
        if self.target_box_index is not None and self.target_box_index < 0:
            raise ValueError("target_box_index must be non-negative")
        if not self.objective_version.strip():
            raise ValueError("objective_version must not be empty")


TrainingObjectiveKind = Literal["clean", "robust_mix", "targeted_repair"]


@dataclass(frozen=True, slots=True)
class TrainingObjective:
    """Versioned weighting contract consumed by training orchestration."""

    kind: TrainingObjectiveKind = "clean"
    version: str = "1.0.0"
    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("version must not be empty")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in self.weights.values()):
            raise ValueError("weights must be finite and non-negative")
