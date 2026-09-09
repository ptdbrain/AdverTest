"""Resolve reviewed catalog datasets to environment-correct runtime paths."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


@dataclass(frozen=True, slots=True)
class CatalogDatasetRuntime:
    """One reviewed catalog dataset and its loader-ready runtime state."""

    name: str
    bundle_name: str
    task_id: str
    root: Path
    manifest_path: Path
    loader_params: dict[str, object]
    anonymized: bool
    sample_count: int | None = None
    blocked_reason: str | None = None

    @property
    def runnable(self) -> bool:
        return self.anonymized and self.blocked_reason is None

    def as_params(self) -> dict[str, object]:
        return dict(self.loader_params)


_SPECS: dict[str, dict[str, Any]] = {
    "kitti": {
        "bundle_name": "kitti2d-100",
        "task_id": "detection2d",
        "local_relative": Path("drive_export_100/detection2d/kitti2d_100"),
        "portable_relative": Path("data/datasets/kitti-demo5"),
        "manifest_field": "manifest_path",
        "extra_params": {"split": "val"},
        "required_dirs": ("image_2", "label_2"),
    },
    "cityscapes_segmentation": {
        "bundle_name": "cityscapes-instance-100",
        "task_id": "segmentation",
        "local_relative": Path("drive_export_100/segmentation/cityscapes_instance_100"),
        "manifest_field": "anonymization_manifest",
        "extra_params": {"split": "val"},
        "required_dirs": ("leftImg8bit", "gtFine"),
    },
    "nuscenes": {
        "bundle_name": "nuscenes-mini-100",
        "task_id": "detection3d",
        "local_relative": Path("drive_export_100/detection3d/nuscenes_mini_100"),
        "manifest_field": "anonymization_manifest",
        "extra_params": {"version": "v1.0-mini", "split": "mini_val"},
        "required_dirs": ("samples", "v1.0-mini"),
    },
}


def resolve_catalog_dataset(name: str, settings: Any) -> CatalogDatasetRuntime | None:
    """Return the reviewed local/worker bundle and validate its gate metadata."""

    spec = _SPECS.get(name)
    if spec is None:
        return None
    data_root = Path(settings.data_root).expanduser().resolve()
    local_root = data_root / spec["local_relative"]
    worker_root = data_root / "catalog" / spec["bundle_name"]
    candidates = [local_root, worker_root]
    if getattr(settings, "bootstrap_portable_demo", False) and "portable_relative" in spec:
        candidates.append(PROJECT_ROOT / spec["portable_relative"])
    root = next((candidate for candidate in candidates if _has_descriptor_and_manifest(candidate)), worker_root)
    manifest = root / "manifest.jsonl"
    params = {
        ("dataroot" if name == "nuscenes" else "root"): str(root),
        spec["manifest_field"]: str(manifest) if name == "nuscenes" else "manifest.jsonl",
        **spec["extra_params"],
    }
    blocked_reason = _validate_bundle(root, spec)
    sample_count = _bundle_sample_count(root)
    return CatalogDatasetRuntime(
        name=name,
        bundle_name=spec["bundle_name"],
        task_id=spec["task_id"],
        root=root,
        manifest_path=manifest,
        loader_params=params,
        anonymized=blocked_reason is None,
        sample_count=sample_count,
        blocked_reason=blocked_reason,
    )


def _has_descriptor_and_manifest(root: Path) -> bool:
    return (root / "dataset.json").is_file() and (root / "manifest.jsonl").is_file()


def _validate_bundle(root: Path, spec: dict[str, Any]) -> str | None:
    descriptor = root / "dataset.json"
    manifest = root / "manifest.jsonl"
    if not descriptor.is_file():
        return "DATASET_DESCRIPTOR_MISSING"
    if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
        return "ANONYMIZATION_MANIFEST_MISSING"
    try:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "DATASET_DESCRIPTOR_INVALID"
    if payload.get("task_id") != spec["task_id"]:
        return "DATASET_TASK_MISMATCH"
    if payload.get("anonymized") is not True:
        return "ANONYMIZATION_REQUIRED"
    missing = [name for name in spec["required_dirs"] if not (root / name).is_dir()]
    if missing:
        return "DATASET_RUNTIME_FILES_MISSING:" + ",".join(missing)
    return None


def _bundle_sample_count(root: Path) -> int | None:
    descriptor = root / "dataset.json"
    if not descriptor.is_file():
        return None
    try:
        value = json.loads(descriptor.read_text(encoding="utf-8")).get("sample_count")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, int) and value > 0 else None
