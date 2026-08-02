"""Group A registration and legacy RNG isolation tests."""

from __future__ import annotations

import random

import numpy as np

from src.attacks import get_attack, load_attacks
from src.attacks.base import AttackContext
from src.core.types import Sample

CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "speckle_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "gaussian_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "spatter",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "saturate",
)


def test_all_group_a_plugins_are_registered() -> None:
    catalog = load_attacks()
    assert all(name in catalog for name in CORRUPTIONS)
    assert len([attack for attack in catalog.values() if attack.group == "A"]) == 19


def test_imagecorruptions_restores_global_rng_state() -> None:
    sample = Sample("corruption", np.full((96, 96, 3), 0.5, dtype=np.float32))
    np.random.seed(123)
    random.seed(456)
    numpy_before = np.random.get_state()
    python_before = random.getstate()

    get_attack("snow").run(
        sample,
        3,
        AttackContext(rng=np.random.default_rng(99)),
    )

    numpy_after = np.random.get_state()
    python_after = random.getstate()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert python_before == python_after
