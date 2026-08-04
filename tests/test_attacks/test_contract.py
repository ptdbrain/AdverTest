"""Shared contract every attack plugin must satisfy.

This file is the safety net for parallel contributions: it is parametrised over
the whole registry, so a new ``src/attacks/**/your_attack.py`` is covered the
moment it is added — nobody has to edit a shared test file.

If one of these fails for your attack, the attack is wrong, not the test:

* severity 0 must be a no-op (sanity check #1 of plan §3)
* stronger severity must mean a stronger effect (sanity check #2)
* the same seed must produce the same pixels (reproducible runs)
* ground truth must never be modified
* the image contract (float32, same shape, values in [0, 1]) must hold
"""

from __future__ import annotations

import numpy as np
import pytest

from src.adapters.base import ModelAdapter
from src.attacks import load_attacks
from src.attacks.base import AttackContext, BaseAttack
from src.core.hashing import array_digest
from src.core.types import GROUP_TITLES, CameraView, LidarFrame, Sample, validate_image

ATTACK_CLASSES = load_attacks().values()
ATTACK_IDS = [attack.name for attack in ATTACK_CLASSES]

pytestmark = pytest.mark.parametrize("attack_cls", ATTACK_CLASSES, ids=ATTACK_IDS)


def _context(attack_cls: type[BaseAttack], adapter: ModelAdapter, seed: int = 7) -> AttackContext:
    """Fresh context; the model is only handed over when the attack declares it."""
    return AttackContext(
        rng=np.random.default_rng(seed),
        model=adapter if attack_cls.needs_model else None,
    )


def _sample_for(attack_cls: type[BaseAttack], sample: Sample) -> Sample:
    if "camera_rig" in attack_cls.required_sensors:
        views = tuple(
            CameraView(name, np.roll(sample.image, index * 2, axis=1), previous_image=sample.image.copy())
            for index, name in enumerate(("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"))
        )
        sample = sample.with_camera_views(views)
    if "lidar" in attack_cls.required_sensors:
        rings = np.arange(32, dtype=np.float32).repeat(8)
        points = np.column_stack((
            np.linspace(1, 40, len(rings), dtype=np.float32),
            np.sin(rings),
            np.cos(rings),
            np.full(len(rings), 1.0, dtype=np.float32),
            rings,
        ))
        sample = sample.with_lidar_frame(LidarFrame(points, sensor_model="HDL32E"))
    return sample


def _distance(sample: Sample, attacked: Sample, attack_cls: type[BaseAttack]) -> float:
    if "lidar" in attack_cls.affected_sensors:
        assert attacked.lidar_frame is not None and sample.lidar_frame is not None
        return float(abs(len(attacked.lidar_frame.points) - len(sample.lidar_frame.points))) + float(
            np.linalg.norm(attacked.lidar_frame.points[:, :4].mean(axis=0) - sample.lidar_frame.points[:, :4].mean(axis=0))
        )
    if "camera_rig" in attack_cls.affected_sensors:
        assert attacked.camera_views
        return float(sum(np.linalg.norm(a.image - b.image) for a, b in zip(attacked.camera_views, sample.camera_views, strict=True)))
    return float(np.linalg.norm(attacked.image - sample.image))


def _attack(attack_cls: type[BaseAttack]) -> BaseAttack:
    """Use a deterministic synthetic artifact only inside shared contract tests."""
    params = (
        {"allow_builtin_patch": True}
        if "allow_builtin_patch" in attack_cls.params_model.model_fields
        else (
            {"depth_policy": "linear_prior"}
            if "depth_policy" in attack_cls.params_model.model_fields
            else {}
        )
    )
    return attack_cls(**params)


def test_catalog_metadata_is_complete(attack_cls: type[BaseAttack]) -> None:
    described = attack_cls.describe()
    assert attack_cls.group in GROUP_TITLES, "group must be one of A..F (plan §2)"
    assert attack_cls.cost_class in {"cheap", "medium", "expensive"}
    assert 1 <= attack_cls.severity_levels <= 10
    assert described["title"], "first docstring line is used as the catalog title"
    assert described["params_schema"]["type"] == "object"
    if attack_cls.needs_gradients:
        assert attack_cls.needs_model, "gradient attacks must also declare needs_model"


def test_severity_zero_is_identity(attack_cls: type[BaseAttack], sample: Sample, adapter: ModelAdapter) -> None:
    source = _sample_for(attack_cls, sample)
    attacked = _attack(attack_cls).run(source, 0, _context(attack_cls, adapter))
    assert array_digest(attacked.image) == array_digest(source.image)


def test_output_respects_image_contract(
    attack_cls: type[BaseAttack],
    sample: Sample,
    adapter: ModelAdapter,
) -> None:
    source = _sample_for(attack_cls, sample)
    attacked = _attack(attack_cls).run(source, 1, _context(attack_cls, adapter))
    validate_image(attacked.image, like=source.image)


def test_ground_truth_is_untouched(
    attack_cls: type[BaseAttack],
    sample: Sample,
    adapter: ModelAdapter,
) -> None:
    source = _sample_for(attack_cls, sample)
    before = array_digest(source.image)
    attacked = _attack(attack_cls).run(
        source,
        attack_cls.severity_levels,
        _context(attack_cls, adapter),
    )
    assert attacked.boxes == source.boxes, "attacks change sensors, never labels"
    assert array_digest(source.image) == before, "the input sample must not be mutated"


def test_same_seed_gives_same_pixels(
    attack_cls: type[BaseAttack],
    sample: Sample,
    adapter: ModelAdapter,
) -> None:
    source = _sample_for(attack_cls, sample)
    first = _attack(attack_cls).run(
        source,
        2,
        _context(attack_cls, adapter, seed=99),
    )
    second = _attack(attack_cls).run(
        source,
        2,
        _context(attack_cls, adapter, seed=99),
    )
    assert array_digest(first.image) == array_digest(second.image)
    if first.lidar_frame is not None:
        assert array_digest(first.lidar_frame.points) == array_digest(second.lidar_frame.points)


def test_attack_actually_changes_the_image(
    attack_cls: type[BaseAttack],
    sample: Sample,
    adapter: ModelAdapter,
) -> None:
    source = _sample_for(attack_cls, sample)
    attacked = _attack(attack_cls).run(source, 1, _context(attack_cls, adapter))
    assert _distance(source, attacked, attack_cls) > 0.0


def test_effect_grows_with_severity(
    attack_cls: type[BaseAttack],
    sample: Sample,
    adapter: ModelAdapter,
) -> None:
    source = _sample_for(attack_cls, sample)
    weak = _attack(attack_cls).run(source, 1, _context(attack_cls, adapter))
    strong = _attack(attack_cls).run(
        source,
        attack_cls.severity_levels,
        _context(attack_cls, adapter),
    )
    assert _distance(source, strong, attack_cls) >= _distance(source, weak, attack_cls), (
        "severity must be ordered: severity 5 cannot perturb less than severity 1 (sanity check #2)"
    )


def test_out_of_range_severity_is_rejected(
    attack_cls: type[BaseAttack],
    sample: Sample,
    adapter: ModelAdapter,
) -> None:
    with pytest.raises(ValueError):
        _attack(attack_cls).run(
            _sample_for(attack_cls, sample),
            attack_cls.severity_levels + 1,
            _context(attack_cls, adapter),
        )


def test_unknown_parameter_is_rejected(attack_cls: type[BaseAttack]) -> None:
    with pytest.raises(Exception, match="extra_inputs_are_not_permitted|Extra inputs"):
        attack_cls(definitely_not_a_parameter=1)
