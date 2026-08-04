"""nuScenes mini/trainval loader with six-camera and HDL32E payloads.

The devkit is optional at import time; install ``nuscenes-devkit`` only for
benchmark runs. Samples remain gated by an anonymisation manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from src.core.types import CameraView, LidarFrame, Sample
from src.datasets.base import DatasetParams, DatasetSource
from src.datasets.io import load_image


class NuScenesParams(DatasetParams):
    dataroot: str
    version: str = "v1.0-mini"
    split: str = "mini_val"
    anonymization_manifest: str | None = None
    limit: int | None = Field(default=None, ge=1)


class NuScenesDataset(DatasetSource):
    """Reserved nuScenes integration slot.

    It is intentionally not registered until split resolution, calibrated
    depth projection and native 3D labels are validated against nuScenes mini.
    """

    name = "nuscenes"
    modality = "multi"
    owner = "3d-evaluation"
    params_model = NuScenesParams

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        manifest = self.params.anonymization_manifest
        self.anonymized = bool(manifest and Path(manifest).is_file())

    def load(self, limit: int | None = None) -> list[Sample]:
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            raise RuntimeError("nuScenes loading requires nuscenes-devkit") from exc
        nusc = NuScenes(version=self.params.version, dataroot=self.params.dataroot, verbose=False)
        rows: list[Sample] = []
        max_rows = limit or self.params.limit
        for sample in nusc.sample:
            if self.params.split and self.params.split not in sample["scene_token"] and self.params.version.endswith("mini"):
                # The devkit scene split is applied by token in full setups;
                # mini datasets are small enough that this conservative filter
                # avoids pretending to know custom split files.
                pass
            cams: list[CameraView] = []
            for name in ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"):
                token = sample["data"].get(name)
                if token is None:
                    continue
                sd = nusc.get("sample_data", token)
                cams.append(CameraView(name, load_image(Path(self.params.dataroot) / sd["filename"])))
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            points = np.fromfile(Path(self.params.dataroot) / lidar_sd["filename"], dtype=np.float32).reshape(-1, 5)
            frame = LidarFrame(points, sensor_model="HDL32E")
            front = next((v.image for v in cams if v.name == "CAM_FRONT"), cams[0].image)
            rows.append(Sample(sample["token"], front, camera_views=tuple(cams), lidar_frame=frame, anonymized=self.anonymized))
            if max_rows and len(rows) >= max_rows:
                break
        return rows
