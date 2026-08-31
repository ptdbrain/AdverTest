"""Explicit bootstrap for a runnable public demo catalog.

This deliberately supports only the server-maintained Ultralytics model ID.
It is not an upload or validation shortcut for user-supplied checkpoints.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.storage.base import ArtifactStorage


def ensure_demo_checkpoint(
    *, enabled: bool, checkpoint_root: str, model_id: str, storage: ArtifactStorage, storage_key: str
) -> Path | None:
    """Fetch the fixed public checkpoint without importing a model framework."""
    if not enabled:
        return None
    target = Path(checkpoint_root).expanduser().resolve() / "surrogates" / f"{model_id}.pt"
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # This fixed storage key is server configuration, never a user-supplied
    # upload.  Avoid importing Ultralytics/PyTorch in the 512 MB API instance.
    payload = storage.get_bytes(storage_key)
    if not payload:
        raise RuntimeError("DEMO_CHECKPOINT_DOWNLOAD_FAILED")
    target.write_bytes(payload)
    return target


def ensure_catalog_checkpoint(*, model_id: str, checkpoint_root: str, storage: ArtifactStorage) -> Path:
    """Fetch one fixed catalog checkpoint into the disposable worker volume."""
    from src.models.catalog import catalog_model

    item = catalog_model(model_id)
    if item is None:
        raise RuntimeError(f"unknown catalog model: {model_id}")
    target = Path(checkpoint_root).expanduser().resolve() / "catalog" / item.filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    payload = storage.get_bytes(item.storage_key)
    if not payload:
        raise RuntimeError(f"catalog checkpoint download failed: {item.storage_key}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def ensure_demo_kitti(
    *, enabled: bool, storage: ArtifactStorage, storage_prefix: str, data_root: str
) -> Path | None:
    """Materialize the reviewed KITTI demo export without weakening its gate."""
    if not enabled:
        return None
    prefix = storage_prefix.rstrip("/") + "/"
    destination = Path(data_root).expanduser().resolve() / "anonymized" / "kitti-de"
    if _is_anonymized_kitti(destination):
        return destination
    if destination.exists():
        raise RuntimeError(f"demo KITTI destination is invalid: {destination}")

    keys = storage.list_keys(prefix)
    if not keys:
        raise RuntimeError(f"no demo KITTI objects found at storage prefix {prefix!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="advertest-kitti-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "kitti-de"
        for key in keys:
            relative = key.removeprefix(prefix)
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise RuntimeError(f"unsafe demo KITTI object key: {key!r}")
            output = staged / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(storage.get_bytes(key))
        if not _is_anonymized_kitti(staged):
            raise RuntimeError("demo KITTI export lacks a valid anonymization descriptor or manifest")
        os.replace(staged, destination)
    return destination


def ensure_demo_catalog(
    *, enabled: bool, storage: ArtifactStorage, storage_prefix: str, data_root: str
) -> Path | None:
    """Materialize the maintained, small demo bundles used by the web catalog.

    The bundles are curated server assets.  They are deliberately separate from
    user imports and keep the normal loaders and anonymisation gates intact.
    """
    if not enabled:
        return None
    prefix = storage_prefix.rstrip("/") + "/"
    destination = Path(data_root).expanduser().resolve() / "demo-catalog"
    marker = destination / ".ready"
    if marker.is_file():
        return destination
    if destination.exists():
        raise RuntimeError(f"demo catalog destination is incomplete: {destination}")
    keys = storage.list_keys(prefix)
    if not keys:
        raise RuntimeError(f"no demo catalog objects found at storage prefix {prefix!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="advertest-catalog-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "demo-catalog"
        for key in keys:
            relative = key.removeprefix(prefix)
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise RuntimeError(f"unsafe demo catalog object key: {key!r}")
            output = staged / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(storage.get_bytes(key))
        if not (staged / ".ready").is_file():
            raise RuntimeError("demo catalog export has no ready marker")
        os.replace(staged, destination)
    return destination


def ensure_anonymized_catalog_bundle(
    *, enabled: bool, storage: ArtifactStorage, storage_prefix: str, data_root: str, bundle_name: str
) -> Path | None:
    """Materialize a reviewed anonymised catalog bundle on a disposable worker.

    Unlike ``ensure_demo_catalog``, this is for a real, versioned benchmark
    bundle and verifies the dataset descriptor before exposing it to a loader.
    """
    if not enabled:
        return None
    prefix = storage_prefix.rstrip("/") + "/"
    destination = Path(data_root).expanduser().resolve() / "catalog" / bundle_name
    if _is_completed_anonymized_bundle(destination):
        return destination
    if destination.exists():
        raise RuntimeError(f"catalog destination is invalid: {destination}")
    keys = storage.list_keys(prefix)
    if not keys:
        raise RuntimeError(f"no catalog objects found at storage prefix {prefix!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="advertest-catalog-bundle-", dir=destination.parent) as temporary:
        staged = Path(temporary) / bundle_name
        for key in keys:
            relative = key.removeprefix(prefix)
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise RuntimeError(f"unsafe catalog object key: {key!r}")
            output = staged / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(storage.get_bytes(key))
        if not _is_completed_anonymized_bundle(staged):
            raise RuntimeError("catalog bundle lacks a completed anonymization descriptor or manifest")
        os.replace(staged, destination)
    return destination


def ensure_drive_export_bundle(
    *, storage: ArtifactStorage, storage_prefix: str, data_root: str, bundle_name: str
) -> Path:
    """Materialize one reviewed 100-sample Drive-export bundle.

    The import is accepted only when its descriptor and manifest agree on the
    fixed 100-sample scope.  This is intentionally separate from legacy
    ``*-200`` catalog validation, whose ``status=complete`` contract was not
    part of the Drive export schema.
    """
    prefix = storage_prefix.rstrip("/") + "/"
    destination = Path(data_root).expanduser().resolve() / "catalog" / bundle_name
    if _is_drive_export_bundle(destination):
        return destination
    if destination.exists():
        raise RuntimeError(f"Drive export destination is invalid: {destination}")
    keys = storage.list_keys(prefix)
    if not keys:
        raise RuntimeError(f"no Drive export objects found at storage prefix {prefix!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="advertest-drive-export-", dir=destination.parent) as temporary:
        staged = Path(temporary) / bundle_name
        for key in keys:
            relative = key.removeprefix(prefix)
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise RuntimeError(f"unsafe Drive export object key: {key!r}")
            output = staged / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(storage.get_bytes(key))
        if not _is_drive_export_bundle(staged):
            raise RuntimeError("Drive export bundle lacks a valid 100-sample descriptor or manifest")
        os.replace(staged, destination)
    return destination


def ensure_drive_export_catalog(*, storage: ArtifactStorage, data_root: str, prefixes: dict[str, str]) -> dict[str, Path]:
    """Materialize the three datasets shipped in the reviewed Drive export."""
    return {
        bundle_name: ensure_drive_export_bundle(
            storage=storage,
            storage_prefix=storage_prefix,
            data_root=data_root,
            bundle_name=bundle_name,
        )
        for bundle_name, storage_prefix in prefixes.items()
    }


def _is_anonymized_kitti(root: Path) -> bool:
    descriptor, manifest = root / "dataset.json", root / "manifest.jsonl"
    if not descriptor.is_file() or not manifest.is_file():
        return False
    try:
        return bool(json.loads(descriptor.read_text(encoding="utf-8")).get("anonymized"))
    except (OSError, json.JSONDecodeError):
        return False


def _is_completed_anonymized_bundle(root: Path) -> bool:
    descriptor, manifest = root / "dataset.json", root / "manifest.jsonl"
    if not descriptor.is_file() or not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
        return False
    try:
        value = json.loads(descriptor.read_text(encoding="utf-8"))
        return bool(value.get("anonymized")) and value.get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _is_drive_export_bundle(root: Path) -> bool:
    descriptor, manifest = root / "dataset.json", root / "manifest.jsonl"
    if not descriptor.is_file() or not manifest.is_file():
        return False
    try:
        payload = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return (
        payload.get("sample_count") == 100
        and len(rows) == 100
        and bool(payload.get("task_id"))
        and payload.get("anonymized") is True
    )
