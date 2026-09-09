"""Dedicated low-memory SAM2.1 worker for the Render Free deployment."""

from __future__ import annotations

import base64
import json
import os
import threading
import zlib
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.adapters.sam2_free_runtime import build_predictor, predict_masks, unload_predictor

app = FastAPI(title="AdverTest SAM2 Free Worker")
_LOCK = threading.Lock()
_PREDICTOR: Any | None = None
_LOADED_CHECKPOINT: str | None = None


class PredictRequest(BaseModel):
    sample_id: str = Field(min_length=1, max_length=256)
    checkpoint_name: str = Field(min_length=1, max_length=256)
    image_base64: str = Field(min_length=1)
    boxes_json: str = Field(min_length=2)


def _checkpoint_paths() -> dict[str, Path]:
    data_root = Path(os.getenv("DATA_ROOT", "data")).expanduser().resolve()
    runs_root = Path(os.getenv("RUNS_ROOT", "runs")).expanduser().resolve()
    return {
        "sam2.1_hiera_small.pt": data_root / "checkpoints" / "sam2" / "sam2.1_hiera_small_fp16.pt",
        "sam21-robust-r1_best.pt": runs_root / "train" / "sam2_r1" / "sam21-robust-r1_best_fp16.pt",
    }


def _config_path() -> Path:
    # ``build_sam2`` resolves this through Hydra's package search path. Passing
    # an absolute filesystem path works accidentally on Windows but is treated
    # as a missing config name on the Linux Render runtime.
    return Path("configs/sam2.1/sam2.1_hiera_s.yaml")


def _get_predictor(checkpoint_name: str) -> Any:
    global _PREDICTOR, _LOADED_CHECKPOINT
    checkpoint = _checkpoint_paths().get(checkpoint_name)
    if checkpoint is None:
        raise HTTPException(status_code=422, detail="CHECKPOINT_NOT_ALLOWED")
    if not checkpoint.is_file():
        raise HTTPException(status_code=409, detail="CHECKPOINT_MISSING")
    if _PREDICTOR is None or _LOADED_CHECKPOINT != checkpoint_name:
        unload_predictor(_PREDICTOR)
        _PREDICTOR = build_predictor(str(checkpoint), str(_config_path()))
        _LOADED_CHECKPOINT = checkpoint_name
    return _PREDICTOR


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "real_model": True,
        "inference_mode": "sam2.1-float16-128px",
        "loaded_checkpoint": _LOADED_CHECKPOINT,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        raw_boxes = json.loads(request.boxes_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="BOXES_JSON_INVALID") from exc
    if not isinstance(raw_boxes, list) or not raw_boxes or len(raw_boxes) > 64:
        raise HTTPException(status_code=422, detail="BOXES_REQUIRED_OR_TOO_MANY")
    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
        from PIL import Image

        with Image.open(BytesIO(image_bytes)) as image:
            image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail="IMAGE_INVALID") from exc

    boxes: list[tuple[float, float, float, float]] = []
    for item in raw_boxes:
        if not isinstance(item, dict) or not isinstance(item.get("coordinates"), list):
            raise HTTPException(status_code=422, detail="BOXES_INVALID")
        coordinates = tuple(float(value) for value in item["coordinates"])
        if len(coordinates) != 4:
            raise HTTPException(status_code=422, detail="BOXES_INVALID")
        boxes.append(coordinates)  # type: ignore[arg-type]

    with _LOCK:
        predictor = _get_predictor(request.checkpoint_name)
        outputs = predict_masks(predictor, image_array, boxes)

    masks = []
    for item, (mask, score) in zip(raw_boxes, outputs, strict=True):
        packed = zlib.compress(np.asarray(mask, dtype=np.uint8).tobytes())
        masks.append(
            {
                "prompt_id": str(item.get("prompt_id", "")),
                "object_id": int(item.get("object_id", -1)),
                "shape": list(mask.shape),
                "mask_zlib_base64": base64.b64encode(packed).decode("ascii"),
                "score": score,
            }
        )
    return {
        "sample_id": request.sample_id,
        "masks": masks,
        "real_model": True,
        "inference_mode": "sam2.1-float16-128px",
    }
