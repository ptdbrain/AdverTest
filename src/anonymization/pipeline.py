"""Resumable KITTI anonymization with content hashes and atomic outputs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.anonymization.detectors import (
    Detection,
    DetectionKind,
    Detector,
    YoloOnnxDetector,
)
from src.core.hashing import file_digest, stable_digest
from src.datasets.io import load_image


class DetectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: str
    confidence: float = Field(default=0.2, gt=0.0, le=1.0)
    iou: float = Field(default=0.5, gt=0.0, le=1.0)
    image_size: int = Field(default=640, ge=320)
    expansion: float = Field(default=0.15, ge=0.0, le=1.0)
    tile_overlap: float = Field(default=0.2, ge=0.0, lt=1.0)


class AnonymizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_dir: str
    output_dir: str
    input_format: Literal["kitti"] = "kitti"
    face_detector: DetectorConfig
    plate_detector: DetectorConfig
    method: Literal["gaussian", "mosaic", "gaussian_mosaic"] = "gaussian_mosaic"
    blur_ratio: float = Field(default=0.25, gt=0.0, le=1.0)
    mosaic_divisor: int = Field(default=8, ge=2)
    limit: int | None = Field(default=None, ge=1)
    sample_ids: list[str] | None = None
    device: str = "cpu"

    @model_validator(mode="after")
    def validate_selection(self) -> AnonymizationConfig:
        if self.limit is not None and self.sample_ids is not None:
            raise ValueError("provide only one of limit or sample_ids")
        if self.sample_ids is not None:
            if not self.sample_ids:
                raise ValueError("sample_ids must not be empty")
            if len(self.sample_ids) != len(set(self.sample_ids)):
                raise ValueError("sample_ids must not contain duplicates")
        return self


@dataclass(frozen=True, slots=True)
class AnonymizationReport:
    root: Path
    processed_samples: int
    resumed_samples: int
    face_detections: int
    plate_detections: int
    status: str = "complete"

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "processed_samples": self.processed_samples,
            "resumed_samples": self.resumed_samples,
            "face_detections": self.face_detections,
            "plate_detections": self.plate_detections,
            "status": self.status,
        }


class DatasetAnonymizer:
    """Anonymize a KITTI folder without changing source files or labels."""

    def __init__(
        self,
        detectors: Mapping[DetectionKind, Detector] | None = None,
    ) -> None:
        self._detectors = dict(detectors) if detectors is not None else None

    def anonymize(self, config: AnonymizationConfig) -> AnonymizationReport:
        source_root = Path(config.input_dir).expanduser().resolve()
        output_root = Path(config.output_dir).expanduser().resolve()
        if source_root == output_root:
            raise ValueError("anonymization output_dir must differ from input_dir")
        images_root = source_root / "image_2"
        labels_root = source_root / "label_2"
        if not images_root.is_dir():
            raise FileNotFoundError(f"KITTI image_2 directory does not exist: {images_root}")

        available_paths = {
            path.stem: path for path in sorted(images_root.glob("*.png"))
        }
        if config.sample_ids is not None:
            missing = sorted(set(config.sample_ids) - available_paths.keys())
            if missing:
                raise ValueError(f"KITTI sample_ids do not exist: {missing}")
            image_paths = [available_paths[sample_id] for sample_id in config.sample_ids]
        else:
            image_paths = list(available_paths.values())
            if config.limit is not None:
                image_paths = image_paths[: config.limit]
        if not image_paths:
            raise ValueError("KITTI input contains no PNG images")

        detectors = self._resolve_detectors(config)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "image_2").mkdir(exist_ok=True)
        (output_root / "label_2").mkdir(exist_ok=True)
        config_payload = config.model_dump(mode="json")
        _write_or_validate_config(output_root / "config.json", config_payload)

        manifest_path = output_root / "manifest.jsonl"
        records = _read_manifest(manifest_path)
        by_sample = {str(record["sample_id"]): record for record in records}
        detector_hashes = {
            kind: detector.checkpoint_hash for kind, detector in detectors.items()
        }
        descriptor = {
            "format": "kitti",
            "status": "in_progress",
            "anonymized": False,
            "source_root": str(source_root),
            "selected_samples": len(image_paths),
            "method": config.method,
            "detectors": {
                kind: {
                    "name": detector.name,
                    "checkpoint_hash": detector.checkpoint_hash,
                }
                for kind, detector in detectors.items()
            },
        }
        _write_json(output_root / "dataset.json", descriptor)

        resumed = 0
        try:
            for image_path in image_paths:
                sample_id = image_path.stem
                source_hash = file_digest(image_path, length=64)
                existing = by_sample.get(sample_id)
                if existing is not None and _record_is_valid(
                    source_root,
                    output_root,
                    existing,
                    source_hash,
                    detector_hashes,
                ):
                    resumed += 1
                    continue

                image = load_image(image_path)
                detections = tuple(
                    detection
                    for detector in detectors.values()
                    for detection in detector.detect(image)
                )
                expanded = tuple(
                    _expand_detection(
                        detection,
                        image.shape,
                        config.face_detector.expansion
                        if detection.kind == "face"
                        else config.plate_detector.expansion,
                    )
                    for detection in detections
                )
                anonymized = _anonymize_regions(image, expanded, config)
                output_image = output_root / "image_2" / image_path.name
                _write_png(output_image, anonymized)

                source_label = labels_root / f"{sample_id}.txt"
                output_label = output_root / "label_2" / f"{sample_id}.txt"
                label_hash = None
                if source_label.is_file():
                    _copy_atomic(source_label, output_label)
                    label_hash = file_digest(output_label, length=64)

                record = {
                    "sample_id": sample_id,
                    "source_path": str(image_path.relative_to(source_root)),
                    "output_path": str(output_image.relative_to(output_root)),
                    "source_hash": source_hash,
                    "output_hash": file_digest(output_image, length=64),
                    "label_path": (
                        str(output_label.relative_to(output_root))
                        if source_label.is_file()
                        else None
                    ),
                    "label_hash": label_hash,
                    "face_count": sum(item.kind == "face" for item in expanded),
                    "plate_count": sum(
                        item.kind == "license_plate" for item in expanded
                    ),
                    "detections": [_detection_payload(item) for item in expanded],
                    "detector_hashes": {
                        kind: detector_hashes[kind] for kind in detectors
                    },
                }
                by_sample[sample_id] = record
                _write_manifest(manifest_path, list(by_sample.values()))
        except Exception as exc:
            descriptor["status"] = "incomplete"
            descriptor["error"] = f"{type(exc).__name__}: {exc}"
            descriptor["processed_samples"] = len(by_sample)
            _write_json(output_root / "dataset.json", descriptor)
            raise

        selected_records = sorted(
            (by_sample[path.stem] for path in image_paths),
            key=lambda record: str(record["sample_id"]),
        )
        face_count = sum(int(record["face_count"]) for record in selected_records)
        plate_count = sum(int(record["plate_count"]) for record in selected_records)
        descriptor.update(
            {
                "status": "complete",
                "anonymized": True,
                "processed_samples": len(selected_records),
                "face_detections": face_count,
                "plate_detections": plate_count,
                "source_fingerprint": stable_digest(
                    [
                        (record["sample_id"], record["source_hash"])
                        for record in selected_records
                    ],
                    length=32,
                ),
                "manifest_hash": stable_digest(selected_records, length=32),
                "review_status": "pending_spot_check",
            }
        )
        _write_json(output_root / "dataset.json", descriptor)
        return AnonymizationReport(
            root=output_root,
            processed_samples=len(selected_records),
            resumed_samples=resumed,
            face_detections=face_count,
            plate_detections=plate_count,
        )

    def _resolve_detectors(
        self,
        config: AnonymizationConfig,
    ) -> dict[DetectionKind, Detector]:
        if self._detectors is not None:
            missing = {"face", "license_plate"} - self._detectors.keys()
            if missing:
                raise ValueError(f"missing anonymization detectors: {sorted(missing)}")
            return self._detectors
        return {
            "face": _build_detector(
                "face",
                config.face_detector,
                device=config.device,
            ),
            "license_plate": _build_detector(
                "license_plate",
                config.plate_detector,
                device=config.device,
            ),
        }


def inspect_anonymized_dataset(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    descriptor_path = root / "dataset.json"
    manifest_path = root / "manifest.jsonl"
    if not descriptor_path.is_file() or not manifest_path.is_file():
        return {"valid": False, "root": str(root), "error": "missing descriptor or manifest"}
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    records = _read_manifest(manifest_path)
    invalid_samples = [
        str(record.get("sample_id"))
        for record in records
        if not _output_record_is_valid(root, record)
    ]
    count_matches = descriptor.get("processed_samples") == len(records)
    manifest_hash_matches = descriptor.get("manifest_hash") == stable_digest(
        records,
        length=32,
    )
    valid = (
        descriptor.get("status") == "complete"
        and descriptor.get("anonymized") is True
        and count_matches
        and manifest_hash_matches
        and not invalid_samples
    )
    return {
        "valid": valid,
        "root": str(root),
        "status": descriptor.get("status"),
        "anonymized": descriptor.get("anonymized"),
        "manifest_records": len(records),
        "count_matches": count_matches,
        "manifest_hash_matches": manifest_hash_matches,
        "invalid_samples": invalid_samples,
        "face_detections": descriptor.get("face_detections", 0),
        "plate_detections": descriptor.get("plate_detections", 0),
        "review_status": descriptor.get("review_status"),
    }


def _build_detector(
    kind: DetectionKind,
    config: DetectorConfig,
    *,
    device: str,
) -> Detector:
    return YoloOnnxDetector(
        kind=kind,
        checkpoint=config.checkpoint,
        confidence=config.confidence,
        iou=config.iou,
        image_size=config.image_size,
        tile_overlap=config.tile_overlap,
        device=device,
    )


def _expand_detection(
    detection: Detection,
    shape: tuple[int, int, int],
    expansion: float,
) -> Detection:
    height, width = shape[:2]
    box_width = max(1.0, detection.x2 - detection.x1)
    box_height = max(1.0, detection.y2 - detection.y1)
    x_pad = box_width * expansion
    y_pad = box_height * expansion
    return Detection(
        kind=detection.kind,
        x1=max(0.0, detection.x1 - x_pad),
        y1=max(0.0, detection.y1 - y_pad),
        x2=min(float(width), detection.x2 + x_pad),
        y2=min(float(height), detection.y2 + y_pad),
        score=detection.score,
        detector=detection.detector,
    )


def _anonymize_regions(
    image: np.ndarray,
    detections: tuple[Detection, ...],
    config: AnonymizationConfig,
) -> np.ndarray:
    canvas = Image.fromarray(np.rint(image * 255.0).astype(np.uint8), mode="RGB")
    for detection in detections:
        box = (
            int(np.floor(detection.x1)),
            int(np.floor(detection.y1)),
            int(np.ceil(detection.x2)),
            int(np.ceil(detection.y2)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        region = canvas.crop(box)
        width, height = region.size
        if config.method in {"mosaic", "gaussian_mosaic"}:
            small = (
                max(1, width // config.mosaic_divisor),
                max(1, height // config.mosaic_divisor),
            )
            region = region.resize(small, Image.Resampling.BILINEAR).resize(
                (width, height),
                Image.Resampling.NEAREST,
            )
        if config.method in {"gaussian", "gaussian_mosaic"}:
            radius = max(2.0, min(width, height) * config.blur_ratio)
            region = region.filter(ImageFilter.GaussianBlur(radius=radius))
        canvas.paste(region, box)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def _detection_payload(detection: Detection) -> dict[str, Any]:
    return {
        "kind": detection.kind,
        "box": [detection.x1, detection.y1, detection.x2, detection.y2],
        "score": detection.score,
        "detector": detection.detector,
    }


def _record_is_valid(
    source_root: Path,
    output_root: Path,
    record: dict[str, Any],
    source_hash: str,
    detector_hashes: dict[DetectionKind, str],
) -> bool:
    if record.get("source_hash") != source_hash:
        return False
    if record.get("detector_hashes") != detector_hashes:
        return False
    source_path = source_root / str(record.get("source_path", ""))
    if not source_path.is_file():
        return False
    return _output_record_is_valid(output_root, record)


def _output_record_is_valid(root: Path, record: dict[str, Any]) -> bool:
    output_path = root / str(record.get("output_path", ""))
    if not output_path.is_file():
        return False
    if file_digest(output_path, length=64) != record.get("output_hash"):
        return False
    label_path = record.get("label_path")
    if label_path is not None:
        resolved_label = root / str(label_path)
        if not resolved_label.is_file():
            return False
        if file_digest(resolved_label, length=64) != record.get("label_hash"):
            return False
    return True


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    ordered = sorted(records, key=lambda record: str(record["sample_id"]))
    content = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        for record in ordered
    )
    _write_text_atomic(path, content)


def _write_or_validate_config(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"output contains a different anonymization config: {path.parent}")
        return
    _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _write_text_atomic(path: Path, content: str) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def _write_png(path: Path, image: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    Image.fromarray(np.rint(image * 255.0).astype(np.uint8), mode="RGB").save(
        temp,
        format="PNG",
    )
    os.replace(temp, path)


def _copy_atomic(source: Path, destination: Path) -> None:
    temp = destination.with_name(f".{destination.name}.tmp")
    temp.write_bytes(source.read_bytes())
    os.replace(temp, destination)
