from __future__ import annotations

import os

import numpy as np
import pytest

from src.adapters import get_adapter
from src.core.objectives import AttackObjective
from src.datasets import get_dataset

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    ("adapter_name", "checkpoint_env", "objective"),
    [
        ("yolo11", "ADVERTEST_YOLO11_CHECKPOINT", AttackObjective()),
        ("faster_rcnn", "ADVERTEST_FRCNN_CHECKPOINT", AttackObjective(kind="dag")),
        (
            "sam2_surrogate",
            "ADVERTEST_SAM2_CHECKPOINT",
            AttackObjective(kind="segmentation_bce"),
        ),
    ],
)
def test_real_surrogate_produces_gpu_gradient(
    adapter_name: str,
    checkpoint_env: str,
    objective: AttackObjective,
) -> None:
    checkpoint = os.getenv(checkpoint_env)
    if not checkpoint:
        pytest.skip(f"set {checkpoint_env} to run this integration test")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    params: dict[str, str] = {"weights": checkpoint, "device": "cuda"}
    if adapter_name == "sam2_surrogate":
        config = os.getenv("ADVERTEST_SAM2_CONFIG")
        if not config:
            pytest.skip("set ADVERTEST_SAM2_CONFIG for the SAM2 integration test")
        params["config"] = config
    adapter = get_adapter(adapter_name, **params)
    sample = get_dataset(
        "synthetic_shapes",
        n_samples=1,
        image_size=64,
        seed=17,
    ).load()[0]
    gradient = adapter.input_gradient(sample, objective)
    assert gradient.shape == sample.image.shape
    assert np.isfinite(gradient).all()
    assert np.any(np.abs(gradient) > 0)
