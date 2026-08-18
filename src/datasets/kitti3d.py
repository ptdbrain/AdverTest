"""KITTI 3D detection source with canonical LiDAR-frame cuboids.

KITTI labels describe cuboids in the rectified camera frame, using a bottom
centre and ``(height, width, length)`` dimensions.  This module converts them
to the platform's LiDAR convention: geometric centre and
``(length, width, height)`` with yaw measured from LiDAR ``+x`` toward ``+y``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np

from src.core.hashing import file_digest
from src.core.types import Box3D, LidarFrame, Modality, Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource
from src.datasets.io import IMAGE_SUFFIXES, load_image
from src.datasets.kitti import DIFFICULTY_LIMITS, LABEL_MAP, Difficulty


@dataclass(frozen=True, slots=True)
class KittiCalibration:
    """KITTI rectification and camera/LiDAR calibration matrices."""

    p2: np.ndarray
    r0_rect: np.ndarray
    tr_velo_to_cam: np.ndarray
    source_path: Path | None = None

    def lidar_to_rectified_camera(self) -> np.ndarray:
        """Return the homogeneous transform from Velodyne to rectified camera."""
        rectification = np.eye(4, dtype=np.float64)
        rectification[:3, :3] = self.r0_rect
        velo_to_camera = np.eye(4, dtype=np.float64)
        velo_to_camera[:3, :4] = self.tr_velo_to_cam
        return rectification @ velo_to_camera

    def rectified_camera_to_lidar(self) -> np.ndarray:
        """Return the inverse transform used for labels in ``label_2``."""
        try:
            return np.linalg.inv(self.lidar_to_rectified_camera())
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"KITTI calibration transform is non-invertible: {self.source_path}") from exc


def load_kitti_calibration(path: Path) -> KittiCalibration:
    """Parse the three calibration records required for KITTI 3D labels."""
    if not path.is_file():
        raise FileNotFoundError(f"KITTI calibration file not found: {path}")
    records: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, values = line.partition(":")
        if not separator:
            continue
        try:
            records[key.strip()] = np.fromstring(values, sep=" ", dtype=np.float64)
        except ValueError as exc:  # pragma: no cover - np.fromstring is permissive
            raise ValueError(f"invalid KITTI calibration record {key!r}: {path}") from exc

    calibration = KittiCalibration(
        p2=_calibration_matrix(records, "P2", (3, 4), path),
        r0_rect=_calibration_matrix(records, "R0_rect", (3, 3), path),
        tr_velo_to_cam=_calibration_matrix(records, "Tr_velo_to_cam", (3, 4), path),
        source_path=path,
    )
    calibration.rectified_camera_to_lidar()
    return calibration


def _calibration_matrix(records: dict[str, np.ndarray], key: str, shape: tuple[int, int], path: Path) -> np.ndarray:
    values = records.get(key)
    if values is None:
        raise ValueError(f"KITTI calibration missing {key}: {path}")
    if values.size != shape[0] * shape[1] or not np.isfinite(values).all():
        raise ValueError(f"KITTI calibration {key} must contain finite {shape[0]}x{shape[1]} values: {path}")
    return values.reshape(shape)


def _normalize_yaw(yaw: float) -> float:
    return float((yaw + np.pi) % (2.0 * np.pi) - np.pi)


def kitti_camera_box_to_lidar(
    *,
    location: tuple[float, float, float],
    height: float,
    width: float,
    length: float,
    rotation_y: float,
    calibration: KittiCalibration,
) -> tuple[float, float, float, float]:
    """Convert one KITTI camera box to LiDAR centre coordinates and yaw.

    ``location`` is the KITTI rectified-camera bottom centre.  Moving it by
    ``-height / 2`` along camera ``y`` gives the geometric centre before the
    inverse ``R0_rect @ Tr_velo_to_cam`` transform is applied.  The KITTI
    longitudinal vector is ``(cos(rotation_y), 0, -sin(rotation_y))``;
    transforming
    that vector gives the LiDAR yaw without relying on an implicit sign rule.
    """
    dimensions = np.asarray((height, width, length), dtype=np.float64)
    location_array = np.asarray(location, dtype=np.float64)
    if not np.isfinite(dimensions).all() or not np.isfinite(location_array).all():
        raise ValueError("KITTI 3D box contains non-finite dimensions or location")
    if np.any(dimensions <= 0.0):
        raise ValueError("KITTI 3D box dimensions must be positive")
    if not np.isfinite(rotation_y):
        raise ValueError("KITTI 3D box rotation_y must be finite")

    camera_to_lidar = calibration.rectified_camera_to_lidar()
    camera_center = np.array(
        (location_array[0], location_array[1] - height / 2.0, location_array[2], 1.0),
        dtype=np.float64,
    )
    lidar_center = camera_to_lidar @ camera_center
    if not np.isclose(lidar_center[3], 1.0):
        lidar_center /= lidar_center[3]

    camera_heading = np.array((np.cos(rotation_y), 0.0, -np.sin(rotation_y)), dtype=np.float64)
    lidar_heading = camera_to_lidar[:3, :3] @ camera_heading
    planar_heading = lidar_heading[:2]
    if np.linalg.norm(planar_heading) <= 1e-12:
        raise ValueError("KITTI calibration maps box heading to a degenerate LiDAR direction")
    return (
        float(lidar_center[0]),
        float(lidar_center[1]),
        float(lidar_center[2]),
        _normalize_yaw(float(np.arctan2(planar_heading[1], planar_heading[0]))),
    )


class Kitti3DParams(DatasetParams):
    """Selection, difficulty, and anonymization policy for a KITTI 3D export."""

    root: str
    split: Literal["train", "val", "all"] = "val"
    difficulty: Difficulty = "moderate"
    sample_ids: tuple[str, ...] | None = None
    anonymization_manifest: str | None = None


@DATASETS.register
class Kitti3D(DatasetSource):
    """KITTI camera-labelled 3D boxes normalized to the Velodyne frame."""

    name: ClassVar[str] = "kitti3d"
    modality: ClassVar[Modality] = "lidar"
    task_id: ClassVar[str] = "detection3d"
    owner: ClassVar[str] = "3d-evaluation"
    loader_version: ClassVar[str] = "kitti-3d-v1"
    params_model: ClassVar[type[DatasetParams]] = Kitti3DParams
    input_schema: ClassVar[tuple[str, ...]] = ("image", "calibration", "lidar_point_cloud")
    annotation_schema: ClassVar[tuple[str, ...]] = ("boxes3d", "class_labels")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        settings: Kitti3DParams = self.params  # type: ignore[assignment]
        self.root = Path(settings.root).expanduser().resolve()
        self.image_dir = self._find_dir("image_2")
        self.velodyne_dir = self._find_dir("velodyne")
        self.calib_dir = self._find_dir("calib")
        self.label_dir = self._find_dir("label_2")
        self.manifest = self._manifest_path(settings.anonymization_manifest)
        self.anonymized = self.manifest is not None and self.manifest.is_file()

    def _find_dir(self, name: str) -> Path:
        for candidate in (self.root / name, self.root / "training" / name):
            if candidate.is_dir():
                return candidate
        return self.root / name

    def _manifest_path(self, configured: str | None) -> Path | None:
        if configured is None:
            return None
        path = Path(configured).expanduser()
        return path if path.is_absolute() else self.root / path

    def info(self) -> DatasetInfo:
        settings: Kitti3DParams = self.params  # type: ignore[assignment]
        return DatasetInfo(
            name=self.name,
            anonymized=self.anonymized,
            modality=self.modality,
            note=f"root={self.root} split={settings.split} difficulty={settings.difficulty}",
        )

    def load(self, limit: int | None = None) -> list[Sample]:
        self._require_layout()
        image_ids = list(self._ids())
        settings: Kitti3DParams = self.params  # type: ignore[assignment]
        if settings.sample_ids is not None:
            available = set(image_ids)
            missing = sorted(set(settings.sample_ids) - available)
            if missing:
                raise ValueError(f"KITTI 3D sample_ids do not exist: {missing}")
            image_ids = list(settings.sample_ids)
        if limit is not None:
            image_ids = image_ids[:limit]
        return [self._load_sample(image_id) for image_id in image_ids]

    def _require_layout(self) -> None:
        required = (self.image_dir, self.velodyne_dir, self.calib_dir, self.label_dir)
        if not all(path.is_dir() for path in required):
            raise FileNotFoundError(
                f"KITTI 3D image_2/velodyne/calib/label_2 directories do not exist under {self.root}"
            )

    def _ids(self) -> Iterable[str]:
        settings: Kitti3DParams = self.params  # type: ignore[assignment]
        if settings.split != "all":
            for split_file in (
                self.root / "ImageSets" / f"{settings.split}.txt",
                self.root / "training" / "ImageSets" / f"{settings.split}.txt",
            ):
                if split_file.is_file():
                    return tuple(
                        line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
                    )
        return tuple(
            sorted(
                path.stem
                for path in self.image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        )

    def _load_sample(self, image_id: str) -> Sample:
        image_path = self._image_path(image_id)
        calibration_path = self.calib_dir / f"{image_id}.txt"
        lidar_path = self.velodyne_dir / f"{image_id}.bin"
        label_path = self.label_dir / f"{image_id}.txt"
        calibration = load_kitti_calibration(calibration_path)
        frame = LidarFrame(
            points=self._load_lidar(lidar_path),
            fields=("x", "y", "z", "intensity"),
            sensor_model="KITTI-Velodyne",
        )
        boxes, dropped = self._read_labels(label_path, calibration)
        return Sample(
            sample_id=f"kitti3d_{image_id}",
            image=load_image(image_path),
            lidar_frame=frame,
            boxes3d=boxes,
            anonymized=self.anonymized,
            meta={
                "dataset": "KITTI",
                "task": "detection3d",
                "coordinate_frame": "LIDAR",
                "box_layout": "x,y,z,length,width,height,yaw",
                "image_id": image_id,
                "source_path": str(lidar_path),
                "source_uri": f"kitti3d://{image_id}",
                "source_format": "kitti",
                "calibration_path": str(calibration_path),
                "loader_version": self.loader_version,
                "split": self.params.split,
                "anonymization_manifest_hash": file_digest(self.manifest, length=64)
                if self.manifest and self.manifest.is_file()
                else None,
                "dropped_labels": dropped,
            },
        )

    def _image_path(self, image_id: str) -> Path:
        candidates = sorted(path for path in self.image_dir.glob(f"{image_id}.*") if path.is_file())
        if not candidates:
            raise FileNotFoundError(f"KITTI image not found: {image_id}")
        return candidates[0]

    @staticmethod
    def _load_lidar(path: Path) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"KITTI point cloud not found: {path}")
        points = np.fromfile(path, dtype=np.float32)
        if points.size % 4 != 0:
            raise ValueError(f"KITTI point cloud must contain x,y,z,intensity tuples: {path}")
        points = points.reshape(-1, 4)
        if not np.isfinite(points).all():
            raise ValueError(f"KITTI point cloud contains NaN or inf: {path}")
        return np.ascontiguousarray(points, dtype=np.float32)

    def _read_labels(self, path: Path, calibration: KittiCalibration) -> tuple[tuple[Box3D, ...], dict[str, int]]:
        if not path.is_file():
            raise FileNotFoundError(f"KITTI label file not found: {path}")
        boxes: list[Box3D] = []
        dropped: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            box = self._parse_label(fields, calibration, dropped)
            if box is not None:
                boxes.append(box)
        return tuple(boxes), dropped

    def _parse_label(self, fields: list[str], calibration: KittiCalibration, dropped: dict[str, int]) -> Box3D | None:
        raw_label = fields[0]
        if len(fields) < 15:
            dropped["malformed"] = dropped.get("malformed", 0) + 1
            return None
        label = LABEL_MAP.get(raw_label)
        if label is None:
            dropped[raw_label] = dropped.get(raw_label, 0) + 1
            return None
        try:
            truncated = float(fields[1])
            occluded = int(float(fields[2]))
            top, bottom = float(fields[5]), float(fields[7])
            height, width, length = (float(value) for value in fields[8:11])
            location = tuple(float(value) for value in fields[11:14])
            rotation_y = float(fields[14])
            x, y, z, yaw = kitti_camera_box_to_lidar(
                location=location,
                height=height,
                width=width,
                length=length,
                rotation_y=rotation_y,
                calibration=calibration,
            )
        except (TypeError, ValueError):
            dropped[f"{raw_label}:invalid_3d"] = dropped.get(f"{raw_label}:invalid_3d", 0) + 1
            return None
        if not self._passes_difficulty(top, bottom, truncated, occluded):
            dropped[f"{raw_label}:difficulty"] = dropped.get(f"{raw_label}:difficulty", 0) + 1
            return None
        return Box3D(x, y, z, length, width, height, yaw, label, native_label=raw_label)

    def _passes_difficulty(self, top: float, bottom: float, truncated: float, occluded: int) -> bool:
        settings: Kitti3DParams = self.params  # type: ignore[assignment]
        min_height, max_occlusion, max_truncation = DIFFICULTY_LIMITS[settings.difficulty]
        return (bottom - top) >= min_height and occluded <= max_occlusion and truncated <= max_truncation
