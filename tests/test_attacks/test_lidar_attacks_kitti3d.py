import numpy as np
import pytest

from src.attacks.base import AttackContext
from src.attacks.corruption.lidar_xyz_noise import LidarXYZNoise
from src.attacks.occlusion.lidar_beam_drop import LidarBeamDrop
from src.attacks.occlusion.lidar_point_dropout import LidarPointDropout
from src.attacks.occlusion.lidar_sector_drop import LidarSectorDrop
from src.attacks.weather.lidar_fog import LidarFog
from src.attacks.weather.lidar_snow import LidarSnow
from src.core.types import Box3D, LidarFrame, Sample


@pytest.fixture
def kitti3d_sample():
    rng = np.random.default_rng(42)
    points = rng.standard_normal((500, 4)).astype(np.float32)
    points[:, :3] *= 20.0  # realistic range
    points[:, 3] = np.abs(points[:, 3])  # positive intensity
    return Sample(
        sample_id="kitti3d-attack-test",
        image=np.zeros((4, 4, 3), dtype=np.float32),
        lidar_frame=LidarFrame(points, ("x", "y", "z", "intensity"), "KITTI-Velodyne"),
        boxes3d=(Box3D(x=10, y=0, z=1, length=4, width=2, height=1.5, yaw=0, label="Car", score=1.0),),
    )


def test_lidar_sector_drop(kitti3d_sample):
    attack = LidarSectorDrop()
    ctx = AttackContext(rng=np.random.default_rng(42))
    res = attack.run(kitti3d_sample, severity=3, ctx=ctx)
    assert res.lidar_frame.points.shape[0] < kitti3d_sample.lidar_frame.points.shape[0]
    assert res.boxes3d is kitti3d_sample.boxes3d
    assert res.image is kitti3d_sample.image
    assert res.lidar_frame.fields == ("x", "y", "z", "intensity")
    assert not np.isnan(res.lidar_frame.points).any()


def test_lidar_fog(kitti3d_sample):
    attack = LidarFog()
    ctx = AttackContext(rng=np.random.default_rng(42))
    res = attack.run(kitti3d_sample, severity=3, ctx=ctx)
    assert res.lidar_frame.points.shape == kitti3d_sample.lidar_frame.points.shape
    assert res.boxes3d is kitti3d_sample.boxes3d
    assert not np.isnan(res.lidar_frame.points).any()


def test_lidar_snow(kitti3d_sample):
    attack = LidarSnow()
    ctx = AttackContext(rng=np.random.default_rng(42))
    res = attack.run(kitti3d_sample, severity=3, ctx=ctx)
    assert res.lidar_frame.points.shape[0] <= kitti3d_sample.lidar_frame.points.shape[0]
    assert res.boxes3d is kitti3d_sample.boxes3d
    assert not np.isnan(res.lidar_frame.points).any()


def test_lidar_beam_drop_raises_value_error(kitti3d_sample):
    attack = LidarBeamDrop()
    ctx = AttackContext(rng=np.random.default_rng(42))
    with pytest.raises(ValueError, match="ring"):
        attack.run(kitti3d_sample, severity=3, ctx=ctx)


def test_lidar_point_dropout(kitti3d_sample):
    attack = LidarPointDropout()
    ctx = AttackContext(rng=np.random.default_rng(42))
    res = attack.run(kitti3d_sample, severity=3, ctx=ctx)
    assert res.lidar_frame.points.shape[0] < kitti3d_sample.lidar_frame.points.shape[0]
    assert res.boxes3d is kitti3d_sample.boxes3d
    assert not np.isnan(res.lidar_frame.points).any()


def test_lidar_xyz_noise(kitti3d_sample):
    attack = LidarXYZNoise()
    ctx = AttackContext(rng=np.random.default_rng(42))
    res = attack.run(kitti3d_sample, severity=3, ctx=ctx)
    assert res.lidar_frame.points.shape == kitti3d_sample.lidar_frame.points.shape
    assert res.boxes3d is kitti3d_sample.boxes3d
    assert not np.isnan(res.lidar_frame.points).any()
