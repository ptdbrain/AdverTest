import numpy as np
import pytest

from src.attacks.base import AttackContext
from src.attacks.occlusion.lidar_point_dropout import LidarPointDropout
from src.core.types import Box3D, LidarFrame, Sample


@pytest.fixture
def kitti_sample():
    rng = np.random.default_rng(42)
    points = rng.standard_normal((1000, 4)).astype(np.float32)
    return Sample(
        sample_id="dropout-test",
        image=np.zeros((4, 4, 3), dtype=np.float32),
        lidar_frame=LidarFrame(points, ("x", "y", "z", "intensity"), "KITTI-Velodyne"),
        boxes3d=(Box3D(x=10, y=0, z=1, length=4, width=2, height=1.5, yaw=0, label="Car"),),
    )

def make_ctx(seed=195):
    return AttackContext(rng=np.random.default_rng(seed))

def test_severity_zero_is_noop(kitti_sample):
    attack = LidarPointDropout()
    result = attack.run(kitti_sample, severity=0, ctx=make_ctx())
    assert result is kitti_sample

def test_severity_one_reduces_points(kitti_sample):
    attack = LidarPointDropout()
    result = attack.run(kitti_sample, severity=1, ctx=make_ctx())
    assert len(result.lidar_frame.points) < len(kitti_sample.lidar_frame.points)

def test_severity_five_stronger_than_one(kitti_sample):
    attack = LidarPointDropout()
    res1 = attack.run(kitti_sample, severity=1, ctx=make_ctx())
    res5 = attack.run(kitti_sample, severity=5, ctx=make_ctx())
    assert len(res5.lidar_frame.points) < len(res1.lidar_frame.points)

def test_same_seed_deterministic(kitti_sample):
    attack = LidarPointDropout()
    res1 = attack.run(kitti_sample, severity=3, ctx=make_ctx(195))
    res2 = attack.run(kitti_sample, severity=3, ctx=make_ctx(195))
    np.testing.assert_array_equal(res1.lidar_frame.points, res2.lidar_frame.points)

def test_gt_boxes_unchanged(kitti_sample):
    attack = LidarPointDropout()
    result = attack.run(kitti_sample, severity=3, ctx=make_ctx())
    assert result.boxes3d is kitti_sample.boxes3d
    assert result.boxes is kitti_sample.boxes

def test_image_unchanged(kitti_sample):
    attack = LidarPointDropout()
    result = attack.run(kitti_sample, severity=3, ctx=make_ctx())
    np.testing.assert_array_equal(result.image, kitti_sample.image)

def test_fields_preserved(kitti_sample):
    attack = LidarPointDropout()
    result = attack.run(kitti_sample, severity=3, ctx=make_ctx())
    assert result.lidar_frame.fields == kitti_sample.lidar_frame.fields

def test_dtype_float32(kitti_sample):
    attack = LidarPointDropout()
    result = attack.run(kitti_sample, severity=3, ctx=make_ctx())
    assert result.lidar_frame.points.dtype == np.float32

def test_all_values_finite(kitti_sample):
    attack = LidarPointDropout()
    result = attack.run(kitti_sample, severity=5, ctx=make_ctx())
    assert np.all(np.isfinite(result.lidar_frame.points))

def test_at_least_one_point_remains(kitti_sample):
    attack = LidarPointDropout()
    result = attack.run(kitti_sample, severity=5, ctx=make_ctx())
    assert len(result.lidar_frame.points) >= 1
