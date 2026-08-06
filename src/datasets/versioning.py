"""Deterministic, manifest-first dataset ingestion."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.hashing import array_digest, stable_digest
from src.core.types import Sample
from src.datasets.base import DatasetSource
from src.datasets.contracts import DatasetVersion, SampleRecord, SplitName


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    logical_source_id: str
    schema_version: str = "1.0.0"
    loader_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetIngestor:
    """Create stable manifests without allowing host paths into identity."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def ingest(
        self,
        source: DatasetSource,
        config: IngestConfig,
    ) -> DatasetVersion:
        loader_version = config.loader_version or source.loader_version
        metadata = _sanitize_identity_metadata(config.metadata)
        records = tuple(
            sorted(
                (
                    _sample_record(
                        sample,
                        logical_source_id=config.logical_source_id,
                        loader_version=loader_version,
                    )
                    for sample in source.load()
                ),
                key=lambda record: record.sample_id,
            )
        )
        identity = {
            "name": config.name,
            "logical_source_id": config.logical_source_id,
            "schema_version": config.schema_version,
            "loader_version": loader_version,
            "records": [record.model_dump(mode="json") for record in records],
            "metadata": metadata,
        }
        manifest_hash = stable_digest(identity, length=64)
        version_id = f"dataset-{stable_digest(identity, length=32)}"
        version = DatasetVersion(
            name=config.name,
            logical_source_id=config.logical_source_id,
            version_id=version_id,
            manifest_hash=manifest_hash,
            schema_version=config.schema_version,
            loader_version=loader_version,
            records=records,
            metadata=metadata,
        )
        destination = self.root / version_id
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination / "manifest.json"
        serialized = version.model_dump_json(indent=2)
        if manifest.exists() and manifest.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"dataset manifest collision for {version_id}")
        if not manifest.exists():
            manifest.write_text(serialized, encoding="utf-8")
        return version


def _sample_record(
    sample: Sample,
    *,
    logical_source_id: str,
    loader_version: str,
) -> SampleRecord:
    source_uri = _logical_source_uri(sample, logical_source_id)
    source_payload = {
        "image": array_digest(sample.image, length=64),
        "depth": array_digest(sample.depth, length=64) if sample.depth is not None else None,
        "mask": array_digest(sample.mask, length=64) if sample.mask is not None else None,
        "lidar": array_digest(sample.lidar, length=64) if sample.lidar is not None else None,
        "camera_views": [
            {
                "name": view.name,
                "image": array_digest(view.image, length=64),
                "depth": array_digest(view.depth, length=64) if view.depth is not None else None,
            }
            for view in sample.camera_views
        ],
    }
    ground_truth_payload = {
        "boxes": [asdict(box) for box in sample.boxes],
        "boxes3d": [asdict(box) for box in sample.boxes3d],
        "mask": array_digest(sample.mask, length=64) if sample.mask is not None else None,
    }
    class_labels = tuple(
        sorted(
            set(sample.meta.get("class_labels", ()))
            | {box.label for box in sample.boxes}
            | {box.label for box in sample.boxes3d}
        )
    )
    split = sample.meta.get("split")
    official_split: SplitName | None = (
        split if split in {"train", "val", "test"} else None
    )
    provenance_keys = (
        "source_format",
        "image_id",
        "native_labels",
        "loader_version",
        "split",
        "anonymization_manifest_hash",
    )
    provenance = {
        key: sample.meta[key]
        for key in provenance_keys
        if key in sample.meta and sample.meta[key] is not None
    }
    provenance["logical_source_id"] = logical_source_id
    provenance["loader_version"] = loader_version
    return SampleRecord(
        sample_id=sample.sample_id,
        source_uri=source_uri,
        source_hash=stable_digest(source_payload, length=64),
        ground_truth_hash=stable_digest(ground_truth_payload, length=64),
        annotation_type=_annotation_type(sample),
        class_labels=class_labels,
        anonymized=sample.anonymized,
        official_split=official_split,
        provenance=provenance,
    )


def _logical_source_uri(sample: Sample, logical_source_id: str) -> str:
    candidate = str(sample.meta.get("source_uri", ""))
    if candidate and "://" in candidate and not candidate.lower().startswith("file://"):
        return candidate
    return f"{logical_source_id}://{sample.sample_id}"


def _annotation_type(sample: Sample) -> str:
    has_boxes = bool(sample.boxes)
    if sample.boxes3d:
        return "boxes3d"
    if has_boxes and sample.mask is not None:
        return "boxes_and_mask"
    if has_boxes:
        return "boxes"
    if sample.mask is not None:
        return "mask"
    return "none"


def _sanitize_identity_metadata(value: Any) -> Any:
    """Drop absolute host paths before hashing or persisting identity metadata."""
    if isinstance(value, dict):
        return {
            str(key): _sanitize_identity_metadata(item)
            for key, item in value.items()
            if not _is_absolute_path(item)
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_identity_metadata(item)
            for item in value
            if not _is_absolute_path(item)
        ]
    return value


def _is_absolute_path(value: Any) -> bool:
    if not isinstance(value, str) or "://" in value:
        return False
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
