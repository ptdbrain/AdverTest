"""Immutable metadata records for locally produced model artefacts."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

TaskKind = Literal["detection2d", "segmentation"]


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """One checkpoint plus the metadata needed to use it reproducibly."""

    id: str
    model_name: str
    task: TaskKind
    checkpoint_path: str | None
    checkpoint_hash: str | None
    parent_id: str | None
    training_metadata: Mapping[str, Any]
    runnable: bool
    blocked_reason: str | None = None


_ROLE_METADATA: dict[str, tuple[str, str | None]] = {
    "yolo_b0": ("yolo11s-kitti-clean-b0", None),
    "yolo_r1": ("yolo11s-kitti-robust-r1", "yolo11s-kitti-clean-b0"),
    "yolo_r2_fog": ("yolo11s-kitti-repaired-r2-fog", "yolo11s-kitti-robust-r1"),
    "yolo_r2_sensor": (
        "yolo11s-kitti-repaired-r2-sensor-fault",
        "yolo11s-kitti-robust-r1",
    ),
}


def scan_yolo_training_runs(root: Path) -> list[ModelVersion]:
    """Discover known Person-B YOLO roles without loading checkpoints.

    The scanner reads only Ultralytics ``args.yaml`` and ``results.csv`` files
    and computes a digest for an existing ``weights/best.pt``. Unknown run
    directory names are intentionally ignored: they have no stable lineage.
    """

    discovered: list[ModelVersion] = []
    for args_path in root.rglob("args.yaml"):
        role = _role_for(args_path, root)
        if role is None:
            continue
        version_id, parent_id = _ROLE_METADATA[role]
        run_root = args_path.parent
        checkpoint = run_root / "weights" / "best.pt"
        metadata = {
            "role": role,
            "args": _read_flat_yaml(args_path),
            "result": _last_csv_row(run_root / "results.csv"),
            "source_run": str(run_root),
        }
        if checkpoint.is_file():
            discovered.append(
                ModelVersion(
                    id=version_id,
                    model_name="yolo11s",
                    task="detection2d",
                    checkpoint_path=str(checkpoint.resolve()),
                    checkpoint_hash=_file_sha256(checkpoint),
                    parent_id=parent_id,
                    training_metadata=metadata,
                    runnable=True,
                )
            )
        else:
            discovered.append(
                ModelVersion(
                    id=version_id,
                    model_name="yolo11s",
                    task="detection2d",
                    checkpoint_path=None,
                    checkpoint_hash=None,
                    parent_id=parent_id,
                    training_metadata=metadata,
                    runnable=False,
                    blocked_reason="CHECKPOINT_MISSING",
                )
            )
    return sorted(discovered, key=lambda version: version.id)


def _role_for(args_path: Path, root: Path) -> str | None:
    try:
        relative = args_path.relative_to(root)
    except ValueError:
        return None
    for part in relative.parts:
        if part in _ROLE_METADATA:
            return part
    return None


def _read_flat_yaml(path: Path) -> dict[str, Any]:
    """Read the scalar fields written by Ultralytics without adding PyYAML."""

    values: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith((" ", "\t", "#")) or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key.strip()] = _scalar(raw.strip())
    return values


def _last_csv_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key.strip(): _scalar((value or "").strip()) for key, value in rows[-1].items()} if rows else {}


def _scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
