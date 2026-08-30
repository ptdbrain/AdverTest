"""Immutable metadata records for locally produced model artefacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config import PROJECT_ROOT

TaskKind = Literal["detection2d", "segmentation", "detection3d"]


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
    # --- Extended lineage fields (Wave 0) ---
    parent_lineage: tuple[str, ...] = ()
    source_training_run: str | None = None
    training_dataset_manifest: str | None = None
    checkpoint_validated: bool = False
    gate_outcome: str | None = None  # "PASSED" | "FAILED" | None
    evidence_tier: str | None = None  # "CPU_CONTRACT_VERIFIED" | "REAL_MODEL_VERIFIED" | ...
    model_family_id: str | None = None
    checkpoint_role: Literal["base", "defence_baseline", "fine_tuned", "repaired"] = "base"


_ROLE_METADATA: dict[str, tuple[str, str | None]] = {
    "yolo_b0": ("yolo11s-kitti-clean-b0", None),
    "yolo_r1": ("yolo11s-kitti-robust-r1", "yolo11s-kitti-clean-b0"),
    "yolo_r2_fog": ("yolo11s-kitti-repaired-r2-fog", "yolo11s-kitti-robust-r1"),
    "yolo_r2_sensor": (
        "yolo11s-kitti-repaired-r2-sensor-fault",
        "yolo11s-kitti-robust-r1",
    ),
}

_PERSON_B_CANONICAL_LINEAGE = {
    "yolo_b0": None,
    "yolo_r1": "yolo11s-clean-b0",
    "yolo_r2_fog": "yolo11s-robust-r1",
    "yolo_r2_sensor": "yolo11s-robust-r1",
}

_ROLE_CHECKPOINT_KIND = {
    "yolo_b0": "defence_baseline",
    "yolo_r1": "fine_tuned",
    "yolo_r2_fog": "repaired",
    "yolo_r2_sensor": "repaired",
}


def scan_yolo_training_runs(root: Path) -> list[ModelVersion]:
    """Discover known Person-B YOLO roles without loading checkpoints.

    The scanner reads only Ultralytics ``args.yaml`` and ``results.csv`` files
    and computes a digest for an existing ``weights/best.pt``. Unknown run
    directory names are intentionally ignored: they have no stable lineage.
    """

    discovered: list[ModelVersion] = _scan_person_b_summaries(root)
    known_ids = {version.id for version in discovered}
    known_roles = {str(version.training_metadata.get("role", "")) for version in discovered}
    for args_path in root.rglob("args.yaml"):
        role = _role_for(args_path, root)
        if role is None:
            continue
        if role in known_roles:
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
                    model_family_id="yolo11",
                    checkpoint_role=_ROLE_CHECKPOINT_KIND[role],
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
                    model_family_id="yolo11",
                    checkpoint_role=_ROLE_CHECKPOINT_KIND[role],
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
        role = summary_path.parent.parent.name
        metadata = {
            "role": role,
            "training_summary": payload,
            "source_run": str(summary_path.parent),
        }
        versions.append(
            ModelVersion(
                id=version_id,
                model_name=str(registration.get("model_id", "yolo11s")),
                task="detection2d",
                checkpoint_path=str(checkpoint.resolve()) if checkpoint.is_file() else None,
                checkpoint_hash=_file_sha256(checkpoint) if checkpoint.is_file() else None,
                parent_id=_PERSON_B_CANONICAL_LINEAGE.get(role, checkpoint_data.get("parent_model_version")),
                training_metadata=metadata,
                runnable=checkpoint.is_file(),
                blocked_reason=None if checkpoint.is_file() else "CHECKPOINT_MISSING",
                model_family_id="yolo11",
                checkpoint_role=_ROLE_CHECKPOINT_KIND.get(role, "fine_tuned"),
            )
        )
    return versions


def scan_model_artifacts(root: Path) -> list[ModelVersion]:
    """Discover runnable YOLO checkpoints and SAM checkpoints awaiting a handoff.

    SAM2 needs more than a weight file: masks, a prompt protocol and an adapter
    are required before scientific metrics can be reported.  A checkpoint is
    still registered, so the UI can tell its owner exactly what is missing.
    """
    effective_root = (
        root
        if root.exists()
        else (PROJECT_ROOT.parent.parent / "runs" if (PROJECT_ROOT.parent.parent / "runs").exists() else root)
    )
    versions = scan_yolo_training_runs(effective_root)
    for checkpoint in effective_root.rglob("sam2*.pt"):
        if not checkpoint.is_file():
            continue
        versions.append(
            ModelVersion(
                id=f"sam2-{checkpoint.stem}",
                model_name="sam2",
                task="segmentation",
                checkpoint_path=str(checkpoint.resolve()),
                checkpoint_hash=_file_sha256(checkpoint),
                parent_id=None,
                training_metadata={"source_checkpoint": str(checkpoint)},
                runnable=False,
                blocked_reason="WAITING_FOR_ARTIFACTS",
                model_family_id="sam2",
            )
        )
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


def list_known_versions() -> list[ModelVersion]:
    """Return all statically-registered model roles as ModelVersion stubs.

    Used by the evidence status endpoint. Does not require filesystem access.
    SAM2 roles are included as WAITING_FOR_ARTIFACTS.
    """
    versions: list[ModelVersion] = []
    parents_by_id = {version_id: parent_id for version_id, parent_id in _ROLE_METADATA.values()}
    for role, (version_id, parent_id) in _ROLE_METADATA.items():
        lineage: list[str] = []
        ancestor_id = parent_id
        while ancestor_id is not None:
            lineage.append(ancestor_id)
            ancestor_id = parents_by_id.get(ancestor_id)
        versions.append(
            ModelVersion(
                id=version_id,
                model_name="yolo11s",
                task="detection2d",
                checkpoint_path=None,
                checkpoint_hash=None,
                parent_id=parent_id,
                training_metadata={"role": role},
                runnable=False,
                blocked_reason="CHECKPOINT_NOT_SCANNED",
                parent_lineage=tuple(lineage),
                model_family_id="yolo11",
                checkpoint_role=_ROLE_CHECKPOINT_KIND[role],
            )
        )
    # SAM2 placeholder — WAITING_FOR_ARTIFACTS
    versions.append(
        ModelVersion(
            id="sam2-base",
            model_name="sam2",
            task="segmentation",
            checkpoint_path=None,
            checkpoint_hash=None,
            parent_id=None,
            training_metadata={"role": "sam2_base"},
            runnable=False,
            blocked_reason="WAITING_FOR_ARTIFACTS",
            model_family_id="sam2",
        )
    )
    return sorted(versions, key=lambda v: v.id)


def scan_base_checkpoints(root: Path) -> list[ModelVersion]:
    """Register only vendor/base weights kept outside training-run lineage."""
    versions: list[ModelVersion] = []
    for model_id in ("yolo11n", "yolo11s"):
        checkpoint = _resolve_surrogate_checkpoint(root, model_id)
        if checkpoint is None:
            continue
        versions.append(
            ModelVersion(
                id=f"{model_id}-base",
                model_name=model_id,
                task="detection2d",
                checkpoint_path=str(checkpoint.resolve()),
                checkpoint_hash=_file_sha256(checkpoint),
                parent_id=None,
                training_metadata={"source": "vendor_base", "role": "base"},
                runnable=True,
                model_family_id="yolo11",
                checkpoint_role="base",
            )
        )

    # 3D PointPillars base checkpoint discovery
    from src.config import PROJECT_ROOT

    is_project_root = root in {PROJECT_ROOT, PROJECT_ROOT / "checkpoints", PROJECT_ROOT.parent.parent / "checkpoints"}

    pp_candidates = [
        root / "pointpillars_kitti_3class.pth",
        root / "pointpillars.pth",
        root / "surrogates" / "pointpillars_kitti_3class.pth",
    ]
    if is_project_root:
        pp_candidates.extend(
            [
                root.parent / "checkpoints" / "pointpillars_kitti_3class.pth",
                root.parent / "data" / "checkpoints" / "pointpillars_kitti_3class.pth",
                PROJECT_ROOT / "checkpoints" / "pointpillars_kitti_3class.pth",
                PROJECT_ROOT.parent.parent / "checkpoints" / "pointpillars_kitti_3class.pth",
            ]
        )
    for pp_ckpt in pp_candidates:
        if pp_ckpt.is_file():
            versions.append(
                ModelVersion(
                    id="pointpillars-kitti-3class-base",
                    model_name="pointpillars-kitti-3class",
                    task="detection3d",
                    checkpoint_path=str(pp_ckpt.resolve()),
                    checkpoint_hash=_file_sha256(pp_ckpt),
                    parent_id=None,
                    training_metadata={
                        "source": "vendor_base",
                        "role": "base",
                        "config_id": "pointpillars-kitti-3class",
                        "config_file": "configs/mmdet3d/pointpillars_hv_secfpn_6x8_160e_kitti-3d-3class.py",
                    },
                    runnable=True,
                    model_family_id="pointpillars3d",
                    checkpoint_role="base",
                )
            )
            break

    # SAM2 Segmentation base checkpoint discovery
    sam2_candidates = [
        root / "sam2" / "sam2.1_hiera_small.pt",
        root / "sam2" / "sam2.1_hiera_s.pt",
        root / "sam2" / "sam2.1_hiera_tiny.pt",
        root / "sam2" / "sam2.1_hiera_t.pt",
        root / "sam2.1_hiera_small.pt",
        root / "sam2_hiera_small.pt",
    ]
    if is_project_root:
        sam2_candidates.extend(
            [
                root.parent / "checkpoints" / "sam2" / "sam2.1_hiera_small.pt",
                root.parent / "checkpoints" / "sam2.1_hiera_small.pt",
                PROJECT_ROOT / "checkpoints" / "sam2" / "sam2.1_hiera_small.pt",
                PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_small.pt",
                PROJECT_ROOT.parent.parent / "checkpoints" / "sam2" / "sam2.1_hiera_small.pt",
                PROJECT_ROOT.parent.parent / "checkpoints" / "sam2.1_hiera_small.pt",
            ]
        )
    for sam2_ckpt in sam2_candidates:
        if sam2_ckpt.is_file():
            versions.append(
                ModelVersion(
                    id="sam2-hiera-small-base",
                    model_name="sam2",
                    task="segmentation",
                    checkpoint_path=str(sam2_ckpt.resolve()),
                    checkpoint_hash=_file_sha256(sam2_ckpt),
                    parent_id=None,
                    training_metadata={
                        "source": "vendor_base",
                        "role": "base",
                        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
                    },
                    runnable=True,
                    model_family_id="sam2",
                    checkpoint_role="base",
                )
            )
            break
    return versions


def _resolve_surrogate_checkpoint(root: Path, model_id: str) -> Path | None:
    local_candidates = (
        root / "surrogates" / f"{model_id}.pt",
        root / f"{model_id}.pt",
    )
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate.resolve()

    from src.config import PROJECT_ROOT

    if root in {PROJECT_ROOT, PROJECT_ROOT / "checkpoints", PROJECT_ROOT.parent.parent / "checkpoints"}:
        project_candidates = (
            root.parent / "checkpoints" / "surrogates" / f"{model_id}.pt",
            root.parent / "checkpoints" / f"{model_id}.pt",
            PROJECT_ROOT / "checkpoints" / "surrogates" / f"{model_id}.pt",
            PROJECT_ROOT / "checkpoints" / f"{model_id}.pt",
            PROJECT_ROOT.parent.parent / "checkpoints" / "surrogates" / f"{model_id}.pt",
            PROJECT_ROOT.parent.parent / "checkpoints" / f"{model_id}.pt",
            PROJECT_ROOT.parent.parent
            / "data"
            / "checkpoints"
            / "uploaded"
            / f"checkpoint-f15d49bc51cd4d27-{model_id}-clean-b0_best.pt",
        )
        for candidate in project_candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None
