"""Task 10: End-to-end acceptance test for PointPillars on KITTI 3D.

Status: WAITING_FOR_GPU_VALIDATION

This test requires:
- CUDA GPU with MMDetection3D installed
- Environment variables: KITTI_ROOT, POINTPILLARS_CONFIG, POINTPILLARS_WEIGHTS
- Real KITTI velodyne .bin files

It is designed to skip cleanly when prerequisites are unavailable.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.attacks.base import AttackContext
from src.attacks.weather.lidar_fog import LidarFog
from src.evaluation.kitti3d import Kitti3DEvaluator
from src.pipeline.protocol import BenchmarkProtocol

_SKIP_REASON = "requires CUDA, MMDetection3D, KITTI data, and PointPillars checkpoint"

_has_cuda = False
try:
    import torch

    _has_cuda = torch.cuda.is_available()
except ImportError:
    pass

_has_env = all(os.environ.get(key) for key in ("KITTI_ROOT", "POINTPILLARS_CONFIG", "POINTPILLARS_WEIGHTS"))


@pytest.mark.models
@pytest.mark.integration
@pytest.mark.skipif(not (_has_cuda and _has_env), reason=_SKIP_REASON)
def test_pointpillars_kitti3d_full_pipeline() -> None:
    """Full E2E: real KITTI → real PointPillars → clean/attacked eval."""
    from src.adapters.pointpillars import PointPillarsAdapter
    from src.datasets.kitti3d import Kitti3D

    # 1. Load real KITTI sample
    dataset = Kitti3D(root=os.environ["KITTI_ROOT"], split="val")
    samples = dataset.load(limit=3)
    assert len(samples) > 0, "no KITTI samples loaded"

    sample = samples[0]

    # 2. Create locked protocol
    sample_ids = tuple(s.sample_id for s in samples[:1])
    protocol = (
        BenchmarkProtocol(
            name="e2e-pointpillars-kitti3d",
            dataset_version_id="kitti3d-e2e-test",
            sample_ids=sample_ids,
            sample_hashes={sid: f"hash-{sid}" for sid in sample_ids},
            ground_truth_hashes={sid: f"gt-{sid}" for sid in sample_ids},
            metric_versions={"kitti_3d_ap": "advertest-bev-v1", "bev_iou": "1.0.0"},
        )
        .transition("VALIDATED")
        .transition("LOCKED")
    )

    # 3. Create PointPillarsAdapter
    adapter = PointPillarsAdapter(
        config=os.environ["POINTPILLARS_CONFIG"],
        weights=os.environ["POINTPILLARS_WEIGHTS"],
    )

    # 4. Clean prediction
    clean_predictions = adapter.predict([sample])
    assert len(clean_predictions) == 1
    assert clean_predictions[0].sample_id == sample.sample_id

    # 5. Clean evaluation
    evaluator = Kitti3DEvaluator()
    clean_result = evaluator.evaluate(clean_predictions, [sample], protocol)

    # 6. Apply lidar_fog severity 3
    attack = LidarFog()
    attack_ctx = AttackContext(rng=np.random.default_rng(195))
    attacked_sample = attack.run(sample, severity=3, ctx=attack_ctx)

    # 7. Verify attack contract
    assert attacked_sample.sample_id == sample.sample_id
    assert attacked_sample.boxes3d == sample.boxes3d  # GT unchanged

    # 8. Attacked prediction
    attacked_predictions = adapter.predict([attacked_sample])
    assert attacked_predictions[0].sample_id == attacked_sample.sample_id

    # 9. Attacked evaluation
    attacked_result = evaluator.evaluate(attacked_predictions, [attacked_sample], protocol)

    # 10. Assertions — do NOT assert attacked < clean for a single sample
    assert clean_result.task == "detection3d"
    assert attacked_result.task == "detection3d"
    assert clean_result.headline.name == "kitti_3d_ap"
    assert attacked_result.headline.name == "kitti_3d_ap"
    assert np.isfinite(clean_result.headline.value)
    assert np.isfinite(attacked_result.headline.value)
    assert clean_result.protocol_id == protocol.protocol_id
    assert attacked_result.protocol_id == protocol.protocol_id
