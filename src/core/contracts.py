"""Versioned wire contracts shared across platform ownership boundaries."""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.types import (
    Box,
    Box3D,
    DetectionPrediction,
    MaskPrediction,
    SegmentationPrediction,
    Task,
)


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class MaskWireV1(_FrozenContract):
    """Portable row-major, zero-first run-length encoding for a boolean mask."""

    encoding: Literal["rle-row-major-zero-first"] = "rle-row-major-zero-first"
    version: Literal["1.0.0"] = "1.0.0"
    shape: tuple[int, int]
    runs: tuple[int, ...]

    @model_validator(mode="after")
    def validate_encoding(self) -> Self:
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("mask shape dimensions must be positive")
        if not self.runs or any(run < 0 for run in self.runs):
            raise ValueError("mask RLE runs must be non-empty and non-negative")
        if any(run == 0 for run in self.runs[1:]):
            raise ValueError("only the initial zero-run may be empty")
        if sum(self.runs) != self.shape[0] * self.shape[1]:
            raise ValueError("mask RLE length does not match shape")
        return self

    @classmethod
    def from_array(cls, mask: np.ndarray) -> MaskWireV1:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"mask must be np.ndarray, got {type(mask).__name__}")
        if mask.ndim != 2 or mask.dtype != np.bool_:
            raise ValueError("mask must be a 2D boolean array")
        flat = mask.reshape(-1)
        runs: list[int] = []
        expected = False
        count = 0
        for value in flat:
            current = bool(value)
            if current == expected:
                count += 1
            else:
                runs.append(count)
                expected = current
                count = 1
        runs.append(count)
        return cls(shape=mask.shape, runs=tuple(runs))

    def to_array(self) -> np.ndarray:
        values = np.concatenate(
            [
                np.full(run, index % 2 == 1, dtype=np.bool_)
                for index, run in enumerate(self.runs)
            ]
        )
        return values.reshape(self.shape)


class BoxWire(_FrozenContract):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_domain(self) -> Box:
        return Box(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            label=self.label,
            score=self.score,
        )


class Box3DWire(_FrozenContract):
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float
    label: str
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    vx: float = 0.0
    vy: float = 0.0
    native_label: str | None = None

    def to_domain(self) -> Box3D:
        return Box3D(**self.model_dump())


class DetectionPredictionWire(_FrozenContract):
    prediction_type: Literal["detection"] = "detection"
    contract_version: Literal["1.0.0"] = "1.0.0"
    sample_id: str
    boxes: tuple[BoxWire, ...] = ()
    boxes3d: tuple[Box3DWire, ...] = ()
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> DetectionPrediction:
        return DetectionPrediction(
            sample_id=self.sample_id,
            boxes=tuple(box.to_domain() for box in self.boxes),
            boxes3d=tuple(box.to_domain() for box in self.boxes3d),
            latency_ms=self.latency_ms,
            metadata=self.metadata,
        )


class MaskPredictionWire(_FrozenContract):
    instance_id: str
    mask: MaskWireV1
    label: str | None = None
    score: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_domain(self) -> MaskPrediction:
        return MaskPrediction(
            instance_id=self.instance_id,
            mask=self.mask.to_array(),
            label=self.label,
            score=self.score,
        )


class SegmentationPredictionWire(_FrozenContract):
    prediction_type: Literal["segmentation"] = "segmentation"
    contract_version: Literal["1.0.0"] = "1.0.0"
    sample_id: str
    instances: tuple[MaskPredictionWire, ...] = ()
    prompt_id: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_domain(self) -> SegmentationPrediction:
        return SegmentationPrediction(
            sample_id=self.sample_id,
            instances=tuple(instance.to_domain() for instance in self.instances),
            prompt_id=self.prompt_id,
            latency_ms=self.latency_ms,
            metadata=self.metadata,
        )


class ModelVersionMetadata(_FrozenContract):
    """Owner-neutral metadata needed to register or compare a model version."""

    model_id: str
    version: str
    task: Task
    framework: str
    checkpoint_hash: str
    preprocessing_version: str
    parent_version: str | None = None
    contract_version: Literal["1.0.0"] = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)


ModelPredictionWire = DetectionPredictionWire | SegmentationPredictionWire
