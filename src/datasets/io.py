"""Canonical image and annotation I/O used by dataset adapters and exporters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.core.types import Box, Box3D, validate_image

IMAGE_SUFFIXES = {".npy", ".png", ".jpg", ".jpeg", ".bmp"}


def load_image(path: Path) -> np.ndarray:
    """Load an image as float32 HWC RGB in [0, 1]."""
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if image.max(initial=0.0) > 1.0:
            image /= 255.0
    else:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("image files require Pillow: pip install Pillow") from exc
        with Image.open(path) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
    image = np.ascontiguousarray(image, dtype=np.float32)
    validate_image(image)
    return image


def load_mask(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("mask image files require Pillow: pip install Pillow") from exc
    with Image.open(path) as opened:
        return np.asarray(opened)


def load_boxes(path: Path | None) -> tuple[Box, ...]:
    if path is None or not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("boxes", []) if isinstance(payload, dict) else payload
    return tuple(
        Box(
            x1=float(row["x1"]),
            y1=float(row["y1"]),
            x2=float(row["x2"]),
            y2=float(row["y2"]),
            label=str(row["label"]),
            score=float(row.get("score", 1.0)),
        )
        for row in rows
    )


def load_boxes3d(path: Path | None) -> tuple[Box3D, ...]:
    """Read optional 3D boxes while keeping v1 image-only labels compatible."""
    if path is None or not path.exists():
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("boxes3d", []) if isinstance(payload, dict) else []
    return tuple(
        Box3D(
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            length=float(row["length"]),
            width=float(row["width"]),
            height=float(row["height"]),
            yaw=float(row["yaw"]),
            label=str(row["label"]),
            score=float(row.get("score", 1.0)),
            vx=float(row.get("vx", 0.0)),
            vy=float(row.get("vy", 0.0)),
            native_label=(str(row["native_label"]) if row.get("native_label") is not None else None),
        )
        for row in rows
    )


def boxes_payload(boxes: tuple[Box, ...]) -> dict[str, list[dict[str, Any]]]:
    return {
        "boxes": [
            {
                "x1": float(box.x1),
                "y1": float(box.y1),
                "x2": float(box.x2),
                "y2": float(box.y2),
                "label": str(box.label),
                "score": float(box.score),
            }
            for box in boxes
        ]
    }


def annotations_payload(
    boxes: tuple[Box, ...],
    boxes3d: tuple[Box3D, ...] = (),
) -> dict[str, list[dict[str, Any]]]:
    """Stable, explicit 2D/3D annotation payload used by generated v2 data."""
    payload: dict[str, list[dict[str, Any]]] = boxes_payload(boxes)
    payload["boxes3d"] = [
        {
            "x": float(box.x),
            "y": float(box.y),
            "z": float(box.z),
            "length": float(box.length),
            "width": float(box.width),
            "height": float(box.height),
            "yaw": float(box.yaw),
            "label": str(box.label),
            "score": float(box.score),
            "vx": float(box.vx),
            "vy": float(box.vy),
            "native_label": box.native_label,
        }
        for box in boxes3d
    ]
    return payload


def find_mask(root: Path, sample_id: str) -> Path | None:
    for suffix in (".npy", ".png", ".bmp"):
        candidate = root / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return None
