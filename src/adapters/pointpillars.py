"""Optional MMDetection3D PointPillars adapter.

The heavy framework is deliberately imported lazily so the core test suite
remains CPU-only. Install the pinned 1.4.0 environment before inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import close, remove
from pathlib import Path
from tempfile import mkstemp
from time import perf_counter
from typing import Any

import numpy as np

from src.adapters.base import ModelAdapter
from src.core.hashing import file_digest
from src.core.types import Box3D, DetectionPrediction, ModelInfo, Sample


class PointPillarsAdapter(ModelAdapter):
    """Reserved MMDetection3D PointPillars integration slot.

    This class is intentionally not registered until ``predict`` supports a
    real checkpoint/config pair, so the model catalog stays executable.
    """

    name = "pointpillars"
    task = "detection3d"
    modality = "lidar"
    version = "mmdet3d-1.4.0"
    runnable = False

    def __init__(
        self,
        *,
        config: str,
        weights: str,
        device: str = "cuda:0",
        score_threshold: float = 0.25,
        max_detections: int = 500,
    ) -> None:
        super().__init__(score_threshold=score_threshold, max_detections=max_detections)
        self.config_path = str(Path(config).expanduser())
        self.weights_path = str(Path(weights).expanduser())
        self.device = device
        self.checkpoint_hash = file_digest(self.weights_path)
        init_model, self._inference_detector = _load_mmdet3d_backend()
        self.model = init_model(self.config_path, self.weights_path, device=self.device)
        self.classes = _model_classes(self.model)

        # These aliases keep the boundary explicit in call sites and make the
        # state shape compatible with existing adapter conventions.
        self._model = self.model
        self._classes = self.classes
        self._checkpoint_hash = self.checkpoint_hash

    def predict(self, samples: Sequence[Sample]) -> list[DetectionPrediction]:
        predictions: list[DetectionPrediction] = []
        for sample in samples:
            if sample.lidar_frame is None:
                raise ValueError(f"PointPillars requires sample.lidar_frame: {sample.sample_id}")
            started = perf_counter()
            point_path = _write_temp_lidar_bin(sample.lidar_frame.points)
            try:
                result = self._inference_detector(self._model, point_path)
            finally:
                remove(point_path)
            predictions.append(
                DetectionPrediction(
                    sample_id=sample.sample_id,
                    boxes3d=tuple(self._boxes_from_result(result)),
                    latency_ms=(perf_counter() - started) * 1000.0,
                    metadata={
                        "coordinate_frame": "LIDAR",
                        "checkpoint_hash": self._checkpoint_hash,
                    },
                )
            )
        return predictions

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            task="detection3d",
            version=self.version,
            modality="lidar",
            supports_gradients=False,
            classes=tuple(self._classes),
            checkpoint_hash=self._checkpoint_hash,
            runnable=False,
        )

    def _boxes_from_result(self, result: Any) -> list[Box3D]:
        instances = result.pred_instances_3d
        tensors = instances.bboxes_3d.tensor.detach().cpu().numpy()
        scores = instances.scores_3d.detach().cpu().numpy()
        labels = instances.labels_3d.detach().cpu().numpy()
        boxes: list[Box3D] = []
        for raw_box, score, label_id in zip(tensors, scores, labels, strict=True):
            if float(score) < self.score_threshold:
                continue
            label_index = int(label_id)
            if not 0 <= label_index < len(self._classes):
                raise ValueError(f"PointPillars returned unknown class index: {label_index}")
            if len(raw_box) < 7:
                raise ValueError("PointPillars returned a 3D box with fewer than seven values")
            # MMDetection3D LiDARInstance3DBoxes documents its tensor as
            # (x, y, z, dx, dy, dz, yaw): dx follows the LiDAR x axis and is
            # canonical length; dy is canonical width.
            x, y, z, length, width, height, yaw = raw_box[:7]
            boxes.append(
                Box3D(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    length=float(length),
                    width=float(width),
                    height=float(height),
                    yaw=float(yaw),
                    label=self._classes[label_index],
                    score=float(score),
                )
            )
            if len(boxes) >= self.max_detections:
                break
        return boxes


def _load_mmdet3d_backend() -> tuple[Any, Any]:
    """Import MMDetection3D only when the optional adapter is instantiated."""
    try:
        from mmdet3d.apis import inference_detector, init_model
    except ImportError as exc:
        raise RuntimeError(
            "PointPillars requires MMDetection3D 1.4.0; install the 3D runtime before inference"
        ) from exc
    return init_model, inference_detector


def _model_classes(model: Any) -> tuple[str, ...]:
    metadata = getattr(model, "dataset_meta", {}) or {}
    classes = metadata.get("classes") if isinstance(metadata, dict) else None
    if classes is None:
        raise RuntimeError("PointPillars model has no dataset_meta.classes label mapping")
    return tuple(str(label) for label in classes)


def _write_temp_lidar_bin(points: np.ndarray) -> str:
    """Write a closed float32 ``.bin`` file so Windows backends can reopen it."""
    descriptor, path = mkstemp(suffix=".bin")
    close(descriptor)
    try:
        np.ascontiguousarray(points, dtype=np.float32).tofile(path)
    except Exception:
        remove(path)
        raise
    return path
