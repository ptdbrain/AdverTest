"""Framework-neutral attack objectives and surrogate capabilities."""

from __future__ import annotations

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
    metadata: dict[str, Any] = field(default_factory=dict)
