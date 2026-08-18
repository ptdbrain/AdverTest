"""CPU contracts for the KITTI 3D LiDAR dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.datasets.kitti3d import Kitti3D


def _write_png(path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(path)


def _write_calibration(path: Path) -> None:
    path.write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        # x_cam=-y_velo, y_cam=-z_velo, z_cam=x_velo
        "Tr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n",
        encoding="utf-8",
    )


def _write_sample(root: Path, image_id: str, *, label: str = "Car", rotation_y: float = 0.0) -> None:
    training = root / "training"
    _write_png(training / "image_2" / f"{image_id}.png")
    np.asarray([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.25]], dtype=np.float32).tofile(
        training / "velodyne" / f"{image_id}.bin"
    )
    _write_calibration(training / "calib" / f"{image_id}.txt")
    (training / "label_2" / f"{image_id}.txt").write_text(
        f"{label} 0 0 0 0 0 50 50 2 2 4 0 0 10 {rotation_y}\n",
        encoding="utf-8",
    )


@pytest.fixture
def kitti3d_root(tmp_path: Path) -> Path:
    training = tmp_path / "training"
    for name in ("image_2", "velodyne", "calib", "label_2"):
        (training / name).mkdir(parents=True)
    _write_sample(tmp_path, "000001")
    _write_sample(tmp_path, "000002", label="Van")
    (tmp_path / "approved.json").write_text("{}\n", encoding="utf-8")
    return tmp_path


def _dataset(root: Path, **params: object) -> Kitti3D:
    return Kitti3D(root=str(root), split="all", anonymization_manifest="approved.json", **params)


def test_kitti3d_loads_lidar_and_boxes(kitti3d_root: Path) -> None:
    sample = _dataset(kitti3d_root).load(limit=1)[0]

    assert sample.lidar_frame is not None
    assert sample.lidar_frame.points.dtype == np.float32
    assert sample.lidar_frame.points.ndim == 2
    assert sample.lidar_frame.points.shape[1] == 4
    assert sample.lidar_frame.fields == ("x", "y", "z", "intensity")
    assert sample.boxes3d


def test_kitti3d_uses_four_lidar_fields(kitti3d_root: Path) -> None:
    frame = _dataset(kitti3d_root).load(limit=1)[0].lidar_frame

    assert frame is not None
    assert frame.fields == ("x", "y", "z", "intensity")
    assert frame.sensor_model == "KITTI-Velodyne"


def test_kitti3d_rejects_malformed_point_cloud(kitti3d_root: Path) -> None:
    np.asarray([1.0, 2.0, 3.0], dtype=np.float32).tofile(kitti3d_root / "training" / "velodyne" / "000001.bin")

    with pytest.raises(ValueError, match="x,y,z,intensity"):
        _dataset(kitti3d_root).load(limit=1)


def test_kitti3d_boxes_have_positive_dimensions(kitti3d_root: Path) -> None:
    boxes = _dataset(kitti3d_root).load(limit=1)[0].boxes3d

    assert all(box.length > 0 and box.width > 0 and box.height > 0 for box in boxes)


def test_kitti3d_uses_supported_classes_only(kitti3d_root: Path) -> None:
    samples = _dataset(kitti3d_root).load()

    assert [box.label for sample in samples for box in sample.boxes3d] == ["Car"]
    assert samples[1].meta["dropped_labels"] == {"Van": 1}


def test_kitti3d_sample_ids_are_deterministic(kitti3d_root: Path) -> None:
    first = _dataset(kitti3d_root).load()
    second = _dataset(kitti3d_root).load()

    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert [sample.meta["image_id"] for sample in first] == ["000001", "000002"]


def test_kitti3d_box_length_axis_yaw_at_zero_rotation_is_lidar(kitti3d_root: Path) -> None:
    sample = _dataset(kitti3d_root).load(limit=1)[0]
    box = sample.boxes3d[0]

    # KITTI labels locate the bottom center in rectified camera coordinates.
    # Their non-square length axis at rotation_y=0 is camera +x.  With this
    # fixture calibration it maps to LiDAR -y, therefore yaw=-pi/2.
    assert (box.x, box.y, box.z) == pytest.approx((10.0, 0.0, 1.0))
    assert box.yaw == pytest.approx(-np.pi / 2.0)
    assert sample.meta["coordinate_frame"] == "LIDAR"
    assert sample.meta["box_layout"] == "x,y,z,length,width,height,yaw"


def test_kitti3d_box_length_axis_yaw_at_nonzero_rotation_is_lidar(kitti3d_root: Path) -> None:
    _write_sample(kitti3d_root, "000001", rotation_y=np.pi / 4.0)

    box = _dataset(kitti3d_root).load(limit=1)[0].boxes3d[0]

    # The KITTI length axis is (cos(ry), 0, -sin(ry)); the fixture maps its
    # ry=pi/4 direction to LiDAR (-sqrt(1/2), -sqrt(1/2), 0).
    assert box.length == pytest.approx(4.0)
    assert box.width == pytest.approx(2.0)
    assert box.yaw == pytest.approx(-3.0 * np.pi / 4.0)


def test_kitti3d_rejects_noninvertible_calibration_transform(kitti3d_root: Path) -> None:
    (kitti3d_root / "training" / "calib" / "000001.txt").write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 0 0 0 0 0 0 0 0 0\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="transform is non-invertible"):
        _dataset(kitti3d_root).load(limit=1)
