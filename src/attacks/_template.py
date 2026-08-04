"""COPY ME — template for a new attack plugin.

Modules starting with ``_`` are skipped by auto-discovery, so this file never
appears in the catalog.

How to add your attack (see ``docs/CONTRIBUTING_ATTACKS.md`` for the full guide):

1. ``cp src/attacks/_template.py src/attacks/<group>/<your_attack>.py``
   (``corruption``=A, ``weather``=B, ``occlusion``=C, ``adversarial``=D,
   ``patch``=E, ``blackbox``=F)
2. Rename the class, set ``name`` (unique, snake_case), ``group``, ``cost_class``,
   ``owner`` (you), ``reference`` (paper/library).
3. Put every tunable number in the params model, one value per severity.
4. Implement ``apply`` — pure function of ``(sample, severity, ctx.rng)``.
5. ``pytest tests/test_attacks -q`` — the shared contract test already covers
   your file. Add ``tests/test_attacks/test_<your_attack>.py`` for the specifics.

Rules that keep the pipeline sane (enforced by ``BaseAttack.run`` + contract test):

* never mutate ``sample``; return ``sample.with_image(new_pixels)``
* never touch ``sample.boxes`` (ground truth stays fixed)
* never use ``np.random`` directly — use ``ctx.rng`` so runs are reproducible
* keep the image shape; values are clipped to ``[0, 1]`` for you
* stronger severity must mean stronger effect (sanity check #2)
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class TemplateParams(AttackParams):
    """One field per tunable; tuples hold the value for severity 1..5."""

    strength_per_severity: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)


@ATTACKS.register
class TemplateAttack(BaseAttack):
    """One-line description shown in the catalog and the report heatmap."""

    name: ClassVar[str] = "template_attack"
    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "your-name"
    reference: ClassVar[str] = "Author et al., Venue Year (arXiv:xxxx.xxxxx)"
    params_model: ClassVar[type[AttackParams]] = TemplateParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        strength = self.level(severity, self.params.strength_per_severity)
        noise = ctx.rng.normal(0.0, strength, size=sample.image.shape).astype(np.float32)
        return sample.with_image(sample.image + noise)
