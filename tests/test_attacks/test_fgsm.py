"""Per-attack test for the white-box reference plugin."""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.base import ModelAdapter
from src.attacks import get_attack
from src.attacks.base import AttackContext, ModelRequiredError
from src.core.types import Sample
from src.evaluation.detection_metrics import average_precision


def test_perturbation_respects_the_epsilon_budget(adapter: ModelAdapter, sample: Sample) -> None:
    attack = get_attack("fgsm")
    for severity, epsilon in enumerate(attack.params.epsilon_per_severity, start=1):
        attacked = attack.run(sample, severity, AttackContext(rng=np.random.default_rng(0), model=adapter))
        linf = float(np.max(np.abs(attacked.image - sample.image)))
        assert linf <= epsilon + 1e-6, f"severity {severity} exceeded its L-inf budget"


def test_running_without_a_model_fails_clearly(sample: Sample) -> None:
    with pytest.raises(ModelRequiredError, match="needs a model adapter"):
        get_attack("fgsm").run(sample, 1, AttackContext(rng=np.random.default_rng(0)))


def test_attack_actually_hurts_detection(adapter: ModelAdapter, samples: list[Sample]) -> None:
    """A white-box attack that does not reduce AP is a broken attack (plan §11).

    Epsilon is raised above the plan default on purpose: the reference model is a
    threshold detector with a much wider decision margin than a real CNN — see
    the module docstring of ``src/attacks/adversarial/fgsm.py``.
    """
    attack = get_attack("fgsm", epsilon_per_severity=(0.02, 0.04, 0.08, 0.16, 0.32))
    context = AttackContext(rng=np.random.default_rng(0), model=adapter)
    attacked = [attack.run(sample, 5, context) for sample in samples]
    clean_ap = average_precision(adapter.predict(samples), samples)
    attacked_ap = average_precision(adapter.predict(attacked), samples)
    assert attacked_ap < clean_ap
