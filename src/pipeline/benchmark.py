"""Benchmark completed attack datasets with one model under test."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.adapters import get_adapter
from src.adapters.base import ModelAdapter
from src.core.hashing import array_digest, file_digest, stable_digest
from src.core.types import Prediction, Sample
from src.datasets import get_dataset
from src.datasets.base import DatasetSource
from src.datasets.io import boxes_payload
from src.evaluation.detection_metrics import (
    DEFAULT_IOU_THRESHOLD,
    average_precision,
    average_precision_per_class,
    detection_attack_success_rate,
    detection_summary,
)
from src.pipeline.generator import (
    AttackGenerationConfig,
    inspect_generated_dataset,
)


class BenchmarkModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "yolo11"
    checkpoint: str | None = None
    device: str = "cpu"
    score_threshold: float = Field(default=0.001, ge=0.0, le=1.0)
    max_detections: int = Field(default=300, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)


class AttackBenchmarkConfig(BaseModel):
    """One model evaluated against one or more immutable generations."""

    model_config = ConfigDict(extra="forbid")

    generation_paths: list[str] = Field(min_length=1)
    model: BenchmarkModelConfig = Field(default_factory=BenchmarkModelConfig)
    iou_threshold: float = Field(
        default=DEFAULT_IOU_THRESHOLD,
        gt=0.0,
        lt=1.0,
    )
    output_dir: str = "data/benchmarks"


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    benchmark_id: str
    root: Path
    report_path: Path
    summary_path: Path
    n_cells: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "root": str(self.root),
            "report_path": str(self.report_path),
            "summary_path": str(self.summary_path),
            "n_cells": self.n_cells,
        }


class AttackDatasetBenchmark:
    """Compare clean and generated pixels without generating new attacks."""

    version = "1.1.0"

    def run(self, config: AttackBenchmarkConfig) -> BenchmarkArtifacts:
        started = perf_counter()
        adapter, checkpoint_hash = self._adapter(config.model)
        model_info = adapter.metadata()
        generation_payloads = [
            self._load_generation(Path(path).expanduser().resolve())
            for path in config.generation_paths
        ]
        benchmark_id = stable_digest(
            {
                "benchmark_version": self.version,
                "config": config.model_dump(mode="json", exclude={"output_dir"}),
                "model_version": model_info.version,
                "checkpoint_hash": checkpoint_hash,
                "generations": [
                    {
                        "generation_id": payload[1]["generation_id"],
                        "manifest_hash": payload[1]["manifest_hash"],
                    }
                    for payload in generation_payloads
                ],
            },
            length=16,
        )
        root = (
            Path(config.output_dir).expanduser().resolve()
            / config.model.name
            / benchmark_id
        )
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "config.json", config.model_dump(mode="json"))

        clean_prediction_cache: dict[str, Prediction] = {}
        cells: list[dict[str, Any]] = []
        for generation_root, descriptor, generation_config, records in generation_payloads:
            clean_source = _source_from_generation(generation_config)
            clean_source.require_anonymized()
            clean_by_id = {
                sample.sample_id: sample
                for sample in clean_source.load(generation_config.limit)
            }
            generated = get_dataset(
                "generated_dataset",
                root=str(generation_root),
            ).load()
            generated_by_id = {sample.sample_id: sample for sample in generated}
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                grouped[int(record["severity"])].append(record)
            for severity, cell_records in sorted(grouped.items()):
                cells.append(
                    self._benchmark_cell(
                        adapter,
                        checkpoint_hash,
                        clean_prediction_cache,
                        generation_root,
                        descriptor,
                        generation_config,
                        severity,
                        cell_records,
                        clean_by_id,
                        generated_by_id,
                        config.iou_threshold,
                    )
                )

        report = {
            "format": "advertest-benchmark-v1",
            "benchmark_id": benchmark_id,
            "benchmark_version": self.version,
            "model": model_info.name,
            "model_version": model_info.version,
            "checkpoint_hash": checkpoint_hash,
            "iou_threshold": config.iou_threshold,
            "score_threshold": config.model.score_threshold,
            "metric": "macro AP at one IoU threshold (AdverTest AP50 when IoU=0.5)",
            "precision_recall_note": (
                "precision/recall counts use model.score_threshold; AP uses the "
                "retained score-ranked detections"
            ),
            "n_cells": len(cells),
            "cells": cells,
            "seconds": round(perf_counter() - started, 3),
            "simulation_only": True,
        }
        report_path = root / "report.json"
        summary_path = root / "summary.csv"
        _write_json(report_path, report)
        _write_summary_csv(summary_path, cells)
        return BenchmarkArtifacts(
            benchmark_id,
            root,
            report_path,
            summary_path,
            len(cells),
        )

    @staticmethod
    def _adapter(
        config: BenchmarkModelConfig,
    ) -> tuple[ModelAdapter, str | None]:
        params = dict(config.params)
        checkpoint_hash: str | None = None
        if config.checkpoint is not None:
            checkpoint = Path(config.checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"benchmark checkpoint does not exist: {checkpoint}"
                )
            checkpoint_hash = file_digest(checkpoint)
            params["weights"] = config.checkpoint
        if config.name in {"yolo11", "faster_rcnn", "sam2_surrogate"} and (
            config.checkpoint is None
        ):
            raise ValueError(
                f"benchmark model {config.name!r} requires an explicit checkpoint"
            )
        params["device"] = config.device
        params["score_threshold"] = config.score_threshold
        params["max_detections"] = config.max_detections
        try:
            adapter = get_adapter(config.name, **params)
        except TypeError:
            params.pop("device", None)
            adapter = get_adapter(config.name, **params)
        if adapter.metadata().task != "detection2d":
            raise ValueError("attack dataset benchmark currently requires detection2d")
        return adapter, checkpoint_hash

    @staticmethod
    def _load_generation(
        root: Path,
    ) -> tuple[Path, dict[str, Any], AttackGenerationConfig, list[dict[str, Any]]]:
        inspected = inspect_generated_dataset(root)
        if not inspected["valid"]:
            raise ValueError(f"generated dataset failed inspection: {root}")
        generation_config = AttackGenerationConfig.model_validate(
            json.loads((root / "config.json").read_text(encoding="utf-8"))
        )
        records = [
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        return root, inspected, generation_config, records

    def _benchmark_cell(
        self,
        adapter: ModelAdapter,
        benchmark_checkpoint_hash: str | None,
        clean_cache: dict[str, Prediction],
        generation_root: Path,
        descriptor: dict[str, Any],
        generation_config: AttackGenerationConfig,
        severity: int,
        records: list[dict[str, Any]],
        clean_by_id: dict[str, Sample],
        generated_by_id: dict[str, Sample],
        iou_threshold: float,
    ) -> dict[str, Any]:
        source_ids = [str(record["source_sample_id"]) for record in records]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                f"generation {descriptor['generation_id']} has duplicate source IDs "
                f"at severity {severity}"
            )
        clean_samples: list[Sample] = []
        attacked_samples: list[Sample] = []
        for record in records:
            source_id = str(record["source_sample_id"])
            variant_id = str(record["variant_id"])
            clean = clean_by_id.get(source_id)
            variant = generated_by_id.get(variant_id)
            if clean is None or variant is None:
                raise ValueError(
                    f"cannot pair source {source_id!r} and variant {variant_id!r}"
                )
            if array_digest(clean.image, length=32) != record["source_hash"]:
                raise ValueError(f"clean source hash changed for {source_id!r}")
            label_hash = stable_digest(boxes_payload(clean.boxes), length=32)
            if label_hash != record["label_hash"]:
                raise ValueError(f"clean label hash changed for {source_id!r}")
            clean_samples.append(clean)
            attacked_samples.append(clean.with_image(variant.image))

        clean_predictions = self._predict_clean_cached(
            adapter,
            clean_samples,
            clean_cache,
        )
        attacked_predictions = adapter.predict(attacked_samples)
        clean_ap = average_precision(clean_predictions, clean_samples, iou_threshold)
        attacked_ap = average_precision(
            attacked_predictions,
            clean_samples,
            iou_threshold,
        )
        degradation = (
            max(0.0, (clean_ap - attacked_ap) / clean_ap)
            if clean_ap > 0.0
            else 0.0
        )
        relative_ap_change = (
            (attacked_ap - clean_ap) / clean_ap if clean_ap > 0.0 else 0.0
        )
        clean_summary = detection_summary(
            clean_predictions,
            clean_samples,
            iou_threshold,
        )
        attacked_summary = detection_summary(
            attacked_predictions,
            clean_samples,
            iou_threshold,
        )
        success = detection_attack_success_rate(
            clean_predictions,
            attacked_predictions,
            clean_samples,
            iou_threshold,
        )
        attack_params = records[0].get("attack_params", {})
        surrogate_hash = _generation_surrogate_hash(records, generation_config)
        return {
            "generation_id": descriptor["generation_id"],
            "generation_root": str(generation_root),
            "source": descriptor["source"],
            "attack": records[0]["attack"],
            "attack_variant": attack_params.get("variant"),
            "attack_version": records[0]["attack_version"],
            "severity": severity,
            "n_samples": len(clean_samples),
            "ap_clean": round(clean_ap, 6),
            "ap_attacked": round(attacked_ap, 6),
            "degradation": round(degradation, 6),
            "relative_ap_change": round(relative_ap_change, 6),
            "ap_clean_per_class": {
                key: round(value, 6)
                for key, value in average_precision_per_class(
                    clean_predictions,
                    clean_samples,
                    iou_threshold,
                ).items()
            },
            "ap_attacked_per_class": {
                key: round(value, 6)
                for key, value in average_precision_per_class(
                    attacked_predictions,
                    clean_samples,
                    iou_threshold,
                ).items()
            },
            "clean": clean_summary.as_dict(),
            "attacked": attacked_summary.as_dict(),
            "false_positive_delta": (
                attacked_summary.false_positives - clean_summary.false_positives
            ),
            "attack_success": success.as_dict(),
            "mean_clean_latency_ms": _mean_latency(clean_predictions),
            "mean_attacked_latency_ms": _mean_latency(attacked_predictions),
            "attack_surrogate_checkpoint_hash": surrogate_hash,
            "benchmark_checkpoint_hash": benchmark_checkpoint_hash,
            "white_box_same_checkpoint": (
                surrogate_hash is not None
                and surrogate_hash == benchmark_checkpoint_hash
            ),
            "patch_artifact_hash": records[0].get("patch_artifact_hash"),
        }

    @staticmethod
    def _predict_clean_cached(
        adapter: ModelAdapter,
        samples: list[Sample],
        cache: dict[str, Prediction],
    ) -> list[Prediction]:
        if not cache and samples:
            adapter.predict(samples[:1])
        pending: list[tuple[str, Sample]] = []
        keys: list[str] = []
        for sample in samples:
            key = stable_digest(
                {
                    "sample_id": sample.sample_id,
                    "image_hash": array_digest(sample.image, length=32),
                    "model": adapter.metadata().version,
                },
                length=32,
            )
            keys.append(key)
            if key not in cache:
                pending.append((key, sample))
        if pending:
            predictions = adapter.predict([sample for _, sample in pending])
            for (key, _), prediction in zip(pending, predictions, strict=True):
                cache[key] = prediction
        return [cache[key] for key in keys]


def _source_from_generation(config: AttackGenerationConfig) -> DatasetSource:
    if config.dataset_name is not None:
        return get_dataset(config.dataset_name, **config.dataset_params)
    return get_dataset(
        "folder_dataset",
        root=config.input_dir,
        input_format=config.input_format,
        anonymization_manifest=config.anonymization_manifest,
    )


def _generation_surrogate_hash(
    records: list[dict[str, Any]],
    config: AttackGenerationConfig,
) -> str | None:
    hashes = {
        str(record["checkpoint_hash"])
        for record in records
        if record.get("checkpoint_hash")
    }
    if len(hashes) > 1:
        raise ValueError("generation contains multiple surrogate checkpoint hashes")
    if hashes:
        return next(iter(hashes))
    patch_path = config.attack_params.get("patch_path")
    if not patch_path:
        return None
    manifest_path = Path(str(patch_path)).expanduser().resolve().parent / "patch-manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_hash = manifest.get("checkpoint_hash")
    return str(checkpoint_hash) if checkpoint_hash else None


def _mean_latency(predictions: list[Prediction]) -> float:
    if not predictions:
        return 0.0
    return round(
        sum(prediction.latency_ms for prediction in predictions) / len(predictions),
        3,
    )


def _write_json(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_summary_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    columns = [
        "generation_id",
        "source",
        "attack",
        "attack_variant",
        "severity",
        "n_samples",
        "ap_clean",
        "ap_attacked",
        "degradation",
        "relative_ap_change",
        "attack_success_rate",
        "clean_precision",
        "clean_recall",
        "attacked_precision",
        "attacked_recall",
        "false_positive_delta",
        "mean_clean_latency_ms",
        "mean_attacked_latency_ms",
        "white_box_same_checkpoint",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                **cell,
                "attack_success_rate": cell["attack_success"]["rate"],
                "clean_precision": cell["clean"]["precision"],
                "clean_recall": cell["clean"]["recall"],
                "attacked_precision": cell["attacked"]["precision"],
                "attacked_recall": cell["attacked"]["recall"],
            }
            for cell in cells
        )
    os.replace(temporary, path)
