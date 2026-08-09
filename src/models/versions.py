"""Immutable metadata records for locally produced model artefacts."""

from __future__ import annotations

import csv
import hashlib
import json
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

    discovered: list[ModelVersion] = _scan_person_b_summaries(root)
    known_ids = {version.id for version in discovered}
    for args_path in root.rglob("args.yaml"):
        role = _role_for(args_path, root)
        if role is None:
            continue
        version_id, parent_id = _ROLE_METADATA[role]
        if version_id in known_ids:
            continue
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


def _scan_person_b_summaries(root: Path) -> list[ModelVersion]:
    """Read B's exported training summaries before legacy Ultralytics folders.

    Those summaries carry the canonical version ID, parent lineage and exported
    checkpoint path.  They prevent punctuation differences in folder roles from
    creating a second model identity for the same trained weight.
    """
    versions: list[ModelVersion] = []
    for summary_path in root.rglob("training_summary.json"):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            registration = payload["registration"]
            checkpoint_data = payload["checkpoint"]
            version_id = str(registration["version_id"])
            relative_checkpoint = Path(str(checkpoint_data["path"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        checkpoint = relative_checkpoint if relative_checkpoint.is_absolute() else root / relative_checkpoint
        # B's portable exports live beside their summary; use that canonical copy
        # when an original Colab-relative path is no longer present.
        if not checkpoint.is_file():
            named = summary_path.parent / relative_checkpoint.name
            checkpoint = named if named.is_file() else checkpoint
        metadata = {
            "role": summary_path.parent.parent.name,
            "training_summary": payload,
            "source_run": str(summary_path.parent),
        }
        versions.append(ModelVersion(
            id=version_id,
            model_name=str(registration.get("model_id", "yolo11s")),
            task="detection2d",
            checkpoint_path=str(checkpoint.resolve()) if checkpoint.is_file() else None,
            checkpoint_hash=_file_sha256(checkpoint) if checkpoint.is_file() else None,
            parent_id=checkpoint_data.get("parent_model_version"),
            training_metadata=metadata,
            runnable=checkpoint.is_file(),
            blocked_reason=None if checkpoint.is_file() else "CHECKPOINT_MISSING",
        ))
    return versions


def scan_model_artifacts(root: Path) -> list[ModelVersion]:
    """Discover runnable YOLO checkpoints and SAM checkpoints awaiting a handoff.

    SAM2 needs more than a weight file: masks, a prompt protocol and an adapter
    are required before scientific metrics can be reported.  A checkpoint is
    still registered, so the UI can tell its owner exactly what is missing.
    """
    versions = scan_yolo_training_runs(root)
    for checkpoint in root.rglob("sam2*.pt"):
        if not checkpoint.is_file():
            continue
        versions.append(ModelVersion(
            id=f"sam2-{checkpoint.stem}", model_name="sam2", task="segmentation",
            checkpoint_path=str(checkpoint.resolve()), checkpoint_hash=_file_sha256(checkpoint),
            parent_id=None, training_metadata={"source_checkpoint": str(checkpoint)},
            runnable=False, blocked_reason="WAITING_FOR_ARTIFACTS",
        ))
    return sorted(versions, key=lambda version: version.id)


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
