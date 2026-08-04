"""Detector contracts and a lazy Ultralytics ONNX implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from src.core.hashing import file_digest

DetectionKind = Literal["face", "license_plate"]


@dataclass(frozen=True, slots=True)
class Detection:
    kind: DetectionKind
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    detector: str


class Detector(Protocol):
    name: str
    checkpoint_hash: str

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        """Return detections in original-image pixel coordinates."""


class YoloOnnxDetector:
    """Run one local YOLO ONNX checkpoint through Ultralytics predict mode."""

    def __init__(
        self,
        *,
        kind: DetectionKind,
        checkpoint: str,
        confidence: float,
        iou: float,
        image_size: int,
        tile_overlap: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.kind = kind
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"{kind} detector checkpoint does not exist: {self.checkpoint}"
            )
        if self.checkpoint.suffix.lower() != ".onnx":
            raise ValueError(
                f"{kind} detector must use an ONNX checkpoint, got: {self.checkpoint}"
            )
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size
        self.tile_overlap = tile_overlap
        self.device = device
        self.name = f"yolo_onnx:{kind}"
        self.checkpoint_hash = file_digest(self.checkpoint, length=64)
        _validate_input_size(self.checkpoint, image_size)
        self._backend: object | None = None

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        backend = self._load()
        pixels = np.rint(image * 255.0).astype(np.uint8)
        detections = list(self._predict(backend, pixels, x_offset=0, y_offset=0))
        if self.tile_overlap > 0:
            height, width = pixels.shape[:2]
            x_starts = _tile_starts(width, self.image_size, self.tile_overlap)
            y_starts = _tile_starts(height, self.image_size, self.tile_overlap)
            for y_start in y_starts:
                for x_start in x_starts:
                    y_end = min(height, y_start + self.image_size)
                    x_end = min(width, x_start + self.image_size)
                    if x_start == 0 and y_start == 0 and x_end == width and y_end == height:
                        continue
                    tile = pixels[y_start:y_end, x_start:x_end]
                    detections.extend(
                        self._predict(
                            backend,
                            tile,
                            x_offset=x_start,
                            y_offset=y_start,
                        )
                    )
        return _non_maximum_suppression(detections, self.iou)

    def _predict(
        self,
        backend: object,
        pixels: np.ndarray,
        *,
        x_offset: int,
        y_offset: int,
    ) -> tuple[Detection, ...]:
        results = backend.predict(  # type: ignore[attr-defined]
            source=pixels,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            coordinates = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            for box, score in zip(coordinates, scores, strict=True):
                detections.append(
                    Detection(
                        kind=self.kind,
                        x1=float(box[0]) + x_offset,
                        y1=float(box[1]) + y_offset,
                        x2=float(box[2]) + x_offset,
                        y2=float(box[3]) + y_offset,
                        score=float(score),
                        detector=self.name,
                    )
                )
        return tuple(detections)

    def _load(self) -> object:
        if self._backend is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "YOLO anonymization requires the models-cpu or models-gpu extra"
                ) from exc
            self._backend = YOLO(str(self.checkpoint), task="detect")
        return self._backend


def _validate_input_size(path: Path, image_size: int) -> None:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "ONNX detector validation requires the models-cpu or models-gpu extra"
        ) from exc
    model = onnx.load(path, load_external_data=False)
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    fixed_height = dimensions[-2].dim_value
    fixed_width = dimensions[-1].dim_value
    if fixed_height and fixed_width and (
        fixed_height != image_size or fixed_width != image_size
    ):
        raise ValueError(
            f"checkpoint input is {fixed_height}x{fixed_width}, "
            f"but image_size is {image_size}: {path}"
        )


def _tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, length - tile_size + 1, step))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _non_maximum_suppression(
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[Detection, ...]:
    kept: list[Detection] = []
    for candidate in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(_iou(candidate, existing) < iou_threshold for existing in kept):
            kept.append(candidate)
    return tuple(kept)


def _iou(left: Detection, right: Detection) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0
