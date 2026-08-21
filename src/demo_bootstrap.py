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


def _is_anonymized_kitti(root: Path) -> bool:
    descriptor, manifest = root / "dataset.json", root / "manifest.jsonl"
    if not descriptor.is_file() or not manifest.is_file():
        return False
    try:
        return bool(json.loads(descriptor.read_text(encoding="utf-8")).get("anonymized"))
    except (OSError, json.JSONDecodeError):
        return False
