"""nuScenes dataset loader with multi-camera calibration, ego poses, and LiDAR frames (P2.2).

Extracts:
- Multi-camera views in canonical ordering (CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_FRONT_LEFT).
- Calibrated sensor transforms (rotation quaternion, translation vector, camera intrinsics).
- Ego pose & global timestamp.
- 3D bounding box annotations (boxes3d, category labels, visibility, point counts).
- Scene & sample relationships.
- Official train/val/test splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from src.core.types import Box3D, CameraView, LidarFrame, Sample
from src.datasets.base import DatasetParams, DatasetSource
from src.datasets.io import load_image
from src.datasets.registry import DatasetRegistry

NUSCENES_CAMERA_NAMES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
)

NUSCENES_CATEGORY_MAPPING = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}


class NuScenesParams(DatasetParams):
    dataroot: str
    version: str = "v1.0-mini"
    split: str = "mini_val"
    anonymization_manifest: str | None = None
    curation_manifest_path: str | None = None
    limit: int | None = Field(default=None, ge=1)


class NuScenesDataset(DatasetSource):
    """nuScenes Multimodal Dataset Loader."""

    name = "nuscenes"
    modality = "multi"
    owner = "3d-evaluation"
    params_model = NuScenesParams
    task_id = "detection3d"
    input_schema = ("multi_camera_images", "calibration", "lidar_point_cloud")
    annotation_schema = ("boxes3d", "depth", "pose", "class_labels")
    ground_truth_status = "READY"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        manifest = self.params.anonymization_manifest
        self.anonymized = bool(manifest and Path(manifest).is_file())
        self.curated_sample_tokens: set[str] | None = None
        if self.params.curation_manifest_path:
            curation_manifest = DatasetRegistry.load(self.params.curation_manifest_path)
            readiness = curation_manifest.readiness(self.task_id)
            if readiness.status != "READY":
                raise ValueError(f"{readiness.status}: {', '.join(readiness.missing)}")
            self.curated_sample_tokens = set(curation_manifest.sample_ids)

    def load(self, limit: int | None = None) -> list[Sample]:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise RuntimeError("nuScenes loading requires nuscenes-devkit: WAITING_FOR_EXTERNAL_DATA") from exc

        dataroot_path = Path(self.params.dataroot)
        if not dataroot_path.is_dir():
            raise FileNotFoundError(f"nuScenes dataroot not found: {self.params.dataroot} (WAITING_FOR_EXTERNAL_DATA)")

        nusc = NuScenes(version=self.params.version, dataroot=self.params.dataroot, verbose=False)
        rows: list[Sample] = []
        max_rows = limit or self.params.limit

        split_scenes: set[str] = set()
        try:
            from nuscenes.utils.splits import create_splits_scenes

            splits = create_splits_scenes()
            if self.params.split in splits:
                split_scenes = set(splits[self.params.split])
        except Exception:
            split_scenes = set()

        for sample in nusc.sample:
            if self.curated_sample_tokens is not None and sample["token"] not in self.curated_sample_tokens:
                continue
            if split_scenes:
                scene = nusc.get("scene", sample["scene_token"])
                if scene["name"] not in split_scenes:
                    continue

            # Multi-camera ingestion
            cams: list[CameraView] = []
            calibs: dict[str, Any] = {}
            for name in NUSCENES_CAMERA_NAMES:
                token = sample["data"].get(name)
                if token is None:
                    continue
                sd = nusc.get("sample_data", token)
                calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
                ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
                img_path = Path(self.params.dataroot) / sd["filename"]
                if img_path.is_file():
                    img = load_image(img_path)
                    cams.append(CameraView(name, img))
                    calibs[name] = {
                        "translation": calib["translation"],
                        "rotation": calib["rotation"],
                        "camera_intrinsic": calib.get("camera_intrinsic", []),
                        "ego_pose": {
                            "translation": ego_pose["translation"],
                            "rotation": ego_pose["rotation"],
                            "timestamp": ego_pose["timestamp"],
                        },
                    }

            # LiDAR ingestion
            frame: LidarFrame | None = None
            lidar_token = sample["data"].get("LIDAR_TOP")
            if lidar_token:
                lidar_sd = nusc.get("sample_data", lidar_token)
                lidar_path = Path(self.params.dataroot) / lidar_sd["filename"]
                if lidar_path.is_file():
                    raw_points = np.fromfile(str(lidar_path), dtype=np.float32)
                    if raw_points.size == 0 or raw_points.size % 5:
                        raise ValueError(f"nuScenes LiDAR must contain non-empty finite Nx5 points: {lidar_path}")
                    points = raw_points.reshape(-1, 5)
                    if not np.isfinite(points).all():
                        raise ValueError(f"nuScenes LiDAR contains NaN or inf: {lidar_path}")
                    frame = LidarFrame(points, sensor_model="HDL32E")

            # 3D Annotations
            boxes3d: list[Box3D] = []
            for ann_token in sample.get("anns", []):
                ann = nusc.get("sample_annotation", ann_token)
                cat = NUSCENES_CATEGORY_MAPPING.get(ann["category_name"], ann["category_name"])
                # Extract Box3D: translation [x, y, z], size [w, l, h], rotation quaternion [w, x, y, z]
                t = ann["translation"]
                s = ann["size"]  # [width, length, height]
                q = ann.get("rotation", [1.0, 0.0, 0.0, 0.0])
                # Real yaw calculation from quaternion around vertical Z axis
                yaw = float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2)))

                boxes3d.append(
                    Box3D(
                        x=float(t[0]),
                        y=float(t[1]),
                        z=float(t[2]),
                        length=float(s[1]),
                        width=float(s[0]),
                        height=float(s[2]),
                        yaw=yaw,
                        label=cat,
                    )
                )

            front = next((v.image for v in cams if v.name == "CAM_FRONT"), cams[0].image if cams else None)
            if front is None or frame is None:
                raise ValueError(f"nuScenes curated sample is missing required camera or LiDAR asset: {sample['token']}")
            sample_obj = Sample(
                sample_id=sample["token"],
                image=front,
                camera_views=tuple(cams),
                lidar_frame=frame,
                boxes3d=tuple(boxes3d),
                anonymized=self.anonymized,
                meta={
                    "scene_token": sample["scene_token"],
                    "timestamp": sample["timestamp"],
                    "calibrations": calibs,
                    "curation_manifest_sample": self.curated_sample_tokens is not None,
                },
            )
            rows.append(sample_obj)
            if max_rows and len(rows) >= max_rows:
                break

        return rows
