"""Immutable per-sample evidence written by a completed test run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw

from src.core.hashing import array_digest
from src.core.types import (
    DetectionPrediction,
    ModelPrediction,
    Sample,
    SegmentationPrediction,
)


class TaskEvidenceSerializer(Protocol):
    """Rendering boundary implemented by the task/model owning team."""

    task: str

    def write(
        self,
        *,
        root: Path,
        clean: Sample,
        attacked: Sample,
        clean_prediction: ModelPrediction,
        attacked_prediction: ModelPrediction,
    ) -> dict[str, str]: ...


def write_task_evidence(
    serializer: TaskEvidenceSerializer,
    *,
    root: str | Path,
    clean: Sample,
    attacked: Sample,
    clean_prediction: ModelPrediction,
    attacked_prediction: ModelPrediction,
) -> dict[str, str]:
    """Delegate boxes/masks/boundaries to the registered task serializer."""
    if clean.sample_id != attacked.sample_id:
        raise ValueError("clean and attacked evidence must refer to the same sample")
    if clean_prediction.sample_id != clean.sample_id:
        raise ValueError("clean prediction sample_id does not match evidence sample")
    if attacked_prediction.sample_id != attacked.sample_id:
        raise ValueError("attacked prediction sample_id does not match evidence sample")
    evidence_root = Path(root).expanduser().resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    return serializer.write(
        root=evidence_root,
        clean=clean,
        attacked=attacked,
        clean_prediction=clean_prediction,
        attacked_prediction=attacked_prediction,
    )


def prediction_payload(prediction: ModelPrediction) -> dict[str, Any]:
    common = {
        "sample_id": prediction.sample_id,
        "latency_ms": round(prediction.latency_ms, 4),
        "metadata": prediction.metadata,
    }
    if isinstance(prediction, DetectionPrediction):
        return {
            "prediction_type": "detection",
            **common,
            "boxes": [
                {
                    "xyxy": list(box.as_tuple()),
                    "label": box.label,
                    "score": round(box.score, 6),
                }
                for box in prediction.boxes
            ],
        }
    if isinstance(prediction, SegmentationPrediction):
        return {
            "prediction_type": "segmentation",
            **common,
            "prompt_id": prediction.prompt_id,
            "instances": [
                {
                    "instance_id": instance.instance_id,
                    "label": instance.label,
                    "score": round(instance.score, 6),
                    "mask_shape": list(instance.mask.shape),
                    "mask_digest": array_digest(instance.mask),
                }
                for instance in prediction.instances
            ],
        }
    raise TypeError(f"unsupported prediction type: {type(prediction).__name__}")


class EvidenceWriter:
    """Write canonical arrays, viewable PNGs and prediction overlays atomically."""

    def __init__(self, root: str) -> None:
        self.root = Path(root).expanduser().resolve()

    def write(
        self,
        *,
        attack: str,
        severity: int,
        clean: Sample,
        attacked: Sample,
        clean_prediction: DetectionPrediction,
        attacked_prediction: DetectionPrediction,
    ) -> dict[str, str]:
        safe_id = clean.sample_id.replace("/", "_").replace("\\", "_")
        root = self.root / attack / f"severity-{severity}" / safe_id
        root.mkdir(parents=True, exist_ok=True)
        clean_npy = root / "clean.npy"
        attacked_npy = root / "attacked.npy"
        np.save(clean_npy, clean.image.astype(np.float32, copy=False))
        np.save(attacked_npy, attacked.image.astype(np.float32, copy=False))
        clean_png = root / "clean.png"
        attacked_png = root / "attacked.png"
        overlay = root / "comparison.png"
        _save_png(clean_png, clean.image)
        _save_png(attacked_png, attacked.image)
        _save_comparison(overlay, clean.image, attacked.image, clean_prediction, attacked_prediction)
        payload = {"clean": prediction_payload(clean_prediction), "attacked": prediction_payload(attacked_prediction)}
        (root / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {
            "clean_image": str(clean_png),
            "attacked_image": str(attacked_png),
            "overlay": str(overlay),
        }


def _save_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _save_comparison(
    path: Path,
    clean: np.ndarray,
    attacked: np.ndarray,
    clean_prediction: DetectionPrediction,
    attacked_prediction: DetectionPrediction,
) -> None:
    left = Image.fromarray(np.rint(np.clip(clean, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
    right = Image.fromarray(np.rint(np.clip(attacked, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
    _draw_boxes(left, clean_prediction, "#00a651")
    _draw_boxes(right, attacked_prediction, "#d92828")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(path)


def _draw_boxes(image: Image.Image, prediction: DetectionPrediction, color: str) -> None:
    draw = ImageDraw.Draw(image)
    for box in prediction.boxes:
        draw.rectangle(box.as_tuple(), outline=color, width=2)
        draw.text((box.x1, max(0, box.y1 - 12)), f"{box.label} {box.score:.2f}", fill=color)
