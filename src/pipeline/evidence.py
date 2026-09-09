"""Immutable per-sample evidence written by a completed test run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from src.core.hashing import array_digest
from src.core.memory import trim_memory
from src.core.types import DetectionPrediction, ModelPrediction, Sample, SegmentationPrediction


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
        payload = {
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
        if getattr(prediction, "boxes3d", None):
            payload["boxes3d"] = [
                {
                    "label": box.label,
                    "score": round(box.score, 6),
                    "x": round(box.x, 4),
                    "y": round(box.y, 4),
                    "z": round(box.z, 4),
                    "length": round(box.length, 4),
                    "width": round(box.width, 4),
                    "height": round(box.height, 4),
                    "yaw": round(box.yaw, 4),
                }
                for box in prediction.boxes3d
            ]
        return payload
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
        clean_prediction: ModelPrediction,
        attacked_prediction: ModelPrediction,
    ) -> dict[str, str]:
        safe_id = clean.sample_id.replace("/", "_").replace("\\", "_")
        root = self.root / attack / f"severity-{severity}" / safe_id
        root.mkdir(parents=True, exist_ok=True)
        import os
        if os.environ.get("SAVE_NUMPY_EVIDENCE", "1").lower() in ("1", "true"):
            clean_npy = root / "clean.npy"
            attacked_npy = root / "attacked.npy"
            np.save(clean_npy, clean.image.astype(np.float32, copy=False))
            np.save(attacked_npy, attacked.image.astype(np.float32, copy=False))
        clean_png = root / "clean.png"
        attacked_png = root / "attacked.png"
        clean_prediction_png = root / "clean-prediction.png"
        attacked_prediction_png = root / "attacked-prediction.png"
        diff_png = root / "diff.png"
        pert_png = root / "perturbation.png"
        _save_png(clean_png, clean.image)
        _save_png(attacked_png, attacked.image)
        _save_prediction(clean_prediction_png, clean.image, clean_prediction, "#00a651")
        _save_prediction(attacked_prediction_png, attacked.image, attacked_prediction, "#d92828")

        diff = np.abs(attacked.image - clean.image)
        _save_png(diff_png, np.clip(diff * 5.0, 0.0, 1.0))
        del diff

        pert = attacked.image - clean.image
        _save_png(pert_png, np.clip(pert * 5.0 + 0.5, 0.0, 1.0))
        del pert

        payload = {"clean": prediction_payload(clean_prediction), "attacked": prediction_payload(attacked_prediction)}
        (root / "predictions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        trim_memory()
        return {
            "clean_image": str(clean_png),
            "attacked_image": str(attacked_png),
            "clean_prediction": str(clean_prediction_png),
            "attacked_prediction": str(attacked_prediction_png),
            "diff_image": str(diff_png),
            "perturbation_image": str(pert_png),
        }


def _save_png(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _save_prediction(
    path: Path,
    image: np.ndarray,
    prediction: ModelPrediction,
    color: str,
) -> None:
    rendered = Image.fromarray(np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
    if isinstance(prediction, SegmentationPrediction):
        rendered = _render_segmentation_prediction(rendered, prediction, color)
    else:
        _draw_boxes(rendered, prediction, color)
    rendered.save(path)


def _render_segmentation_prediction(
    image: Image.Image, prediction: SegmentationPrediction, color: str
) -> Image.Image:
    """Overlay the actual predicted masks; never substitute a placeholder mask."""
    rgba = image.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    rgb = ImageColor.getrgb(color)
    for instance in prediction.instances:
        mask = np.asarray(instance.mask, dtype=np.bool_)
        if mask.shape != (rgba.height, rgba.width):
            continue
        mask_image = Image.fromarray(np.where(mask, 150, 0).astype(np.uint8), mode="L")
        layer = Image.new("RGBA", rgba.size, (*rgb, 96))
        overlay.paste(layer, (0, 0), mask_image)
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _draw_boxes(image: Image.Image, prediction: DetectionPrediction, color: str) -> None:
    draw = ImageDraw.Draw(image)
    for box in prediction.boxes:
        draw.rectangle(box.as_tuple(), outline=color, width=2)
        draw.text((box.x1, max(0, box.y1 - 12)), f"{box.label} {box.score:.2f}", fill=color)
