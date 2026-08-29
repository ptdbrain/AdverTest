"""P2.1 - P2.5 Sensor Validation, NuScenes Loader, Beam Drop, Frame Freeze, and PGD Tests.

Covers:
- P2.1: LiDAR .bin and .pcd format validation & ingestion guards (empty, truncated, NaN/Inf, out-of-range).
- P2.2: NuScenes loader schema, canonical camera ordering, and WAITING_FOR_EXTERNAL_DATA handling.
- P2.3: LiDAR beam drop deterministic ring inference and provenance.
- P2.4: Multi-camera synchronous frame freeze on sequence frames and first-frame policy.
- P2.5: PGD gradient fidelity and adversarial group classification.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.attacks.adversarial.pgd import Pgd
from src.attacks.base import AttackContext
from src.attacks.occlusion.frame_freeze import FrameFreeze
from src.attacks.occlusion.lidar_beam_drop import LidarBeamDrop, infer_lidar_rings
from src.core.types import CameraView, LidarFrame, Sample
from src.datasets.lidar_validation import (
    LidarValidationError,
    validate_lidar_bin_bytes,
    validate_pcd_content,
)
from src.datasets.nuscenes import NUSCENES_CAMERA_NAMES, NuScenesDataset

# ==========================================
# P2.1 LiDAR Validation Tests
# ==========================================

def test_lidar_bin_validation_valid() -> None:
    """P2.1: Valid float32 .bin pointcloud should parse cleanly."""
    points = np.array(
        [[10.0, 5.0, 0.5, 0.8], [12.0, 6.0, 0.3, 0.5], [15.0, -2.0, -0.1, 0.9]],
        dtype=np.float32,
    )
    raw_bytes = points.tobytes()
    parsed = validate_lidar_bin_bytes(raw_bytes, min_points=3, coord_range=(-100.0, 100.0))
    assert parsed.shape == (3, 4)
    assert np.allclose(parsed, points)


def test_lidar_bin_validation_rejections() -> None:
    """P2.1: Rejections for empty, truncated, NaN, out-of-range, and below-min points."""
    # 1. Empty
    with pytest.raises(LidarValidationError) as exc:
        validate_lidar_bin_bytes(b"")
    assert exc.value.code == "LIDAR_EMPTY"

    # 2. Truncated (not divisible by 16)
    with pytest.raises(LidarValidationError) as exc:
        validate_lidar_bin_bytes(b"1234567")
    assert exc.value.code == "LIDAR_TRUNCATED"

    # 3. Point count below minimum
    single_pt = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32).tobytes()
    with pytest.raises(LidarValidationError) as exc:
        validate_lidar_bin_bytes(single_pt, min_points=5)
    assert exc.value.code == "LIDAR_POINT_COUNT_BELOW_MINIMUM"

    # 4. NaN / Inf
    nan_pt = np.array([[float("nan"), 2.0, 3.0, 0.5]], dtype=np.float32).tobytes()
    with pytest.raises(LidarValidationError) as exc:
        validate_lidar_bin_bytes(nan_pt, min_points=1)
    assert exc.value.code == "LIDAR_NAN_INF"

    # 5. Out of range coordinates
    oor_pt = np.array([[10000.0, 2.0, 3.0, 0.5]], dtype=np.float32).tobytes()
    with pytest.raises(LidarValidationError) as exc:
        validate_lidar_bin_bytes(oor_pt, min_points=1, coord_range=(-100.0, 100.0))
    assert exc.value.code == "LIDAR_COORDINATES_OUT_OF_RANGE"


def test_pcd_validation_valid_and_invalid() -> None:
    """P2.1: ASCII PCD parsing and validation."""
    valid_pcd = (
        "# .PCD v0.7\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        "WIDTH 2\n"
        "HEIGHT 1\n"
        "POINTS 2\n"
        "DATA ascii\n"
        "1.0 2.0 3.0\n"
        "4.0 5.0 6.0\n"
    )
    parsed = validate_pcd_content(valid_pcd, min_points=2)
    assert parsed.shape == (2, 3)

    invalid_pcd = "VERSION 0.7\nDATA ascii\n1.0\n"
    with pytest.raises(LidarValidationError):
        validate_pcd_content(invalid_pcd, min_points=1)


# ==========================================
# P2.2 nuScenes Loader Tests
# ==========================================

def test_nuscenes_contract_and_camera_names() -> None:
    """P2.2: Canonical 6-camera ordering and dataset status."""
    assert len(NUSCENES_CAMERA_NAMES) == 6
    assert NUSCENES_CAMERA_NAMES[0] == "CAM_FRONT"
    assert NUSCENES_CAMERA_NAMES[-1] == "CAM_FRONT_LEFT"

    dataset = NuScenesDataset(dataroot="/nonexistent/nuscenes/path")
    with pytest.raises((RuntimeError, FileNotFoundError)) as exc:
        dataset.load()
    assert "WAITING_FOR_EXTERNAL_DATA" in str(exc.value)


# ==========================================
# P2.3 Beam Drop Tests
# ==========================================

def test_lidar_beam_drop_deterministic_inference() -> None:
    """P2.3: Deterministic ring inference and provenance tracking."""
    # Synthetic sphere with varied elevation angles
    angles = np.linspace(-math.pi / 4, math.pi / 4, 100)
    points = np.column_stack([np.cos(angles) * 10, np.zeros(100), np.sin(angles) * 10, np.ones(100)]).astype(
        np.float32
    )

    rings = infer_lidar_rings(points, num_rings=16)
    assert len(rings) == 100
    assert len(np.unique(rings)) > 1

    sample = Sample(
        "test-lidar",
        np.zeros((32, 32, 3), dtype=np.float32),
        lidar_frame=LidarFrame(points, fields=("x", "y", "z", "intensity")),
    )
    attack = LidarBeamDrop(allow_inferred_ring=True)
    ctx = AttackContext(rng=np.random.default_rng(2026))

    attacked = attack.apply(sample, severity=3, ctx=ctx)
    assert len(attacked.lidar_frame.points) < len(points)
    assert attacked.meta.get("provenance", {}).get("inferred_ring") is True
    assert attacked.meta.get("provenance", {}).get("ring_inference_algorithm") == "inclination_clustering_v1"


# ==========================================
# P2.4 Frame Freeze Tests
# ==========================================

def test_multi_camera_sequence_frame_freeze() -> None:
    """P2.4: Frame freeze on multi-camera 3-timestamp sequence."""
    f0_cams = (
        CameraView("CAM_FRONT", np.full((16, 16, 3), 0.1, dtype=np.float32)),
        CameraView("CAM_BACK", np.full((16, 16, 3), 0.15, dtype=np.float32)),
    )
    f1_cams = (
        CameraView("CAM_FRONT", np.full((16, 16, 3), 0.5, dtype=np.float32)),
        CameraView("CAM_BACK", np.full((16, 16, 3), 0.55, dtype=np.float32)),
    )
    f2_cams = (
        CameraView("CAM_FRONT", np.full((16, 16, 3), 0.9, dtype=np.float32)),
        CameraView("CAM_BACK", np.full((16, 16, 3), 0.95, dtype=np.float32)),
    )

    sample_0 = Sample("s0", f0_cams[0].image, camera_views=f0_cams, meta={"timestamp": 100, "frame_index": 0})
    sample_1 = Sample("s1", f1_cams[0].image, camera_views=f1_cams, meta={"timestamp": 200, "frame_index": 1})
    sample_2 = Sample("s2", f2_cams[0].image, camera_views=f2_cams, meta={"timestamp": 300, "frame_index": 2})

    history = [sample_0, sample_1, sample_2]
    sample_0.meta["sequence_history"] = history
    sample_1.meta["sequence_history"] = history
    sample_2.meta["sequence_history"] = history

    attack = FrameFreeze()
    ctx = AttackContext(rng=np.random.default_rng(42))

    # 1. At frame 0: cannot freeze initial frame
    with pytest.raises(ValueError) as exc:
        attack.apply(sample_0, severity=1, ctx=ctx)
    assert "FIRST_FRAME_IN_SEQUENCE_CANNOT_FREEZE" in str(exc.value)

    # 2. At frame 1 (severity 1 -> stale 1 -> replays frame 0)
    frozen_1 = attack.apply(sample_1, severity=1, ctx=ctx)
    assert np.allclose(frozen_1.camera_views[0].image, f0_cams[0].image)
    assert np.allclose(frozen_1.camera_views[1].image, f0_cams[1].image)
    assert frozen_1.meta["provenance"]["source_frame_id"] == "s0"
    assert frozen_1.meta["provenance"]["source_timestamp"] == 100

    # 3. At frame 2 (severity 2 -> stale 2 -> replays frame 0)
    frozen_2 = attack.apply(sample_2, severity=2, ctx=ctx)
    assert np.allclose(frozen_2.camera_views[0].image, f0_cams[0].image)
    assert frozen_2.meta["provenance"]["source_frame_id"] == "s0"


# ==========================================
# P2.5 PGD Fidelity Tests
# ==========================================

def test_pgd_gradient_fidelity_and_group() -> None:
    """P2.5: PGD must be classified under Group D and require gradients."""
    attack = Pgd()
    assert attack.group == "D"
    assert attack.needs_gradients is True
    assert attack.needs_model is True
    assert "input_gradient" in attack.required_capabilities
