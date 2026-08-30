"""Immutable local-dataset manifests and readiness checks.

The registry records source paths and selected IDs; it never copies, moves, or
silently expands a source dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetReadiness:
    status: str
    missing: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    task_id: str
    modalities: tuple[str, ...]
    sample_ids: tuple[str, ...]
    source_root: str | None = None
    provenance: dict[str, Any] | None = None
    limitations: tuple[str, ...] = ()

    def readiness(self, task_id: str) -> DatasetReadiness:
        requirements = {
            "detection2d": ("image_2", "label_2"),
            "segmentation": ("image_2", "semantic"),
            "detection3d": ("image_2", "label_2", "calib", "velodyne"),
        }
        required = requirements.get(task_id, ())
        if task_id == "detection3d" and {"lidar", "calibration", "ego_pose", "annotations"}.issubset(self.modalities):
            required = ()
        missing = tuple(item for item in required if item not in self.modalities)
        if task_id != self.task_id:
            missing = (*missing, "task_id")
        if missing:
            return DatasetReadiness("INSUFFICIENT_FOR_REQUESTED_SUBSET", missing, self.limitations)
        if not self.sample_ids:
            return DatasetReadiness("INSUFFICIENT_FOR_REQUESTED_SUBSET", ("sample_ids",), self.limitations)
        return DatasetReadiness("READY", limitations=self.limitations)


class DatasetRegistry:
    @staticmethod
    def load(path: str | Path) -> DatasetManifest:
        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"INVALID_DATASET_MANIFEST: {manifest_path}") from exc
        for key in ("dataset_id", "task_id", "modalities"):
            if not payload.get(key):
                raise ValueError(f"INVALID_DATASET_MANIFEST: missing {key}")
        return DatasetManifest(
            dataset_id=str(payload["dataset_id"]),
            task_id=str(payload["task_id"]),
            modalities=tuple(str(value) for value in payload["modalities"]),
            sample_ids=tuple(str(value) for value in payload.get("sample_ids", ())),
            source_root=payload.get("source_root"),
            provenance=payload.get("provenance"),
            limitations=tuple(str(value) for value in payload.get("limitations", ())),
        )
