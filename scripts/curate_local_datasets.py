"""Create auditable, non-destructive subset manifests for local datasets.

The default operation only inspects a data root and writes manifests when the
CLI is explicitly pointed at an output directory.  ``--apply`` moves only the
previously enumerated, unselected source assets into a same-volume quarantine;
``--finalize`` is the separate irreversible step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"CURATION_TARGET_OUTSIDE_DATA_ROOT: {path}") from exc


def _asset_records(root: Path, paths: Iterable[Path]) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for path in sorted({candidate.resolve() for candidate in paths}):
        if path.is_file():
            records.append({"path": _relative_to(root, path), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    return tuple(records)


def _removal_records(root: Path, paths: Iterable[Path]) -> tuple[dict[str, Any], ...]:
    """List exact destructive candidates without paying a full-dataset hash cost.

    The content hash is captured immediately after the same-volume move in the
    quarantine journal, which is the immutable object used by finalization.
    """
    records: list[dict[str, Any]] = []
    for path in sorted({candidate.resolve() for candidate in paths}):
        if path.is_file():
            records.append({"path": _relative_to(root, path), "size_bytes": path.stat().st_size})
    return tuple(records)


def _stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.iterdir() if path.is_file()}


def _first_existing(root: Path, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    return root / candidates[0]


def _stable_rank(value: str) -> tuple[str, str]:
    return (hashlib.sha256(f"p195-curation-v1:{value}".encode()).hexdigest(), value)


def _select_ids(ids: Iterable[str], count: int) -> tuple[str, ...]:
    return tuple(sorted(sorted(ids, key=_stable_rank)[:count]))


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    dataset_id: str
    task_id: str
    source_root: str
    modalities: tuple[str, ...]
    retained_ids: tuple[str, ...]
    missing_pairs: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    retained_assets: tuple[dict[str, Any], ...] = ()
    removal_assets: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CurationPlan:
    data_root: str
    count: int
    datasets: dict[str, DatasetPlan]
    schema_version: str = "p195-curation-v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "data_root": self.data_root,
            "count": self.count,
            "datasets": {name: asdict(dataset) for name, dataset in self.datasets.items()},
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> CurationPlan:
        datasets = {
            name: DatasetPlan(
                dataset_id=str(value["dataset_id"]),
                task_id=str(value["task_id"]),
                source_root=str(value["source_root"]),
                modalities=tuple(value.get("modalities", ())),
                retained_ids=tuple(value.get("retained_ids", ())),
                missing_pairs=tuple(value.get("missing_pairs", ())),
                limitations=tuple(value.get("limitations", ())),
                retained_assets=tuple(value.get("retained_assets", ())),
                removal_assets=tuple(value.get("removal_assets", ())),
                provenance=dict(value.get("provenance", {})),
            )
            for name, value in dict(payload.get("datasets", {})).items()
        }
        return CurationPlan(
            data_root=str(payload["data_root"]),
            count=int(payload["count"]),
            datasets=datasets,
            schema_version=str(payload.get("schema_version", "p195-curation-v1")),
        )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    missing_pairs: tuple[str, ...]
    bad_hashes: tuple[str, ...]
    dangling_tokens: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not (self.missing_pairs or self.bad_hashes or self.dangling_tokens)

    def as_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, **asdict(self)}


def _kitti_plan(data_root: Path, count: int) -> DatasetPlan:
    source_root = _first_existing(data_root, ("kitti/Kitti/raw/training", "kitti/training"))
    image_dir = source_root / "image_2"
    label_dir = source_root / "label_2"
    image_ids, label_ids = _stems(image_dir), _stems(label_dir)
    paired = image_ids & label_ids
    retained_ids = _select_ids(paired, count)
    missing = tuple(sorted(image_ids ^ label_ids))
    retained_paths = [path for sample_id in retained_ids for path in (next(iter(image_dir.glob(f"{sample_id}.*")), image_dir / f"{sample_id}.png"), label_dir / f"{sample_id}.txt")]
    removal_paths = [path for directory in (image_dir, label_dir) for path in directory.iterdir() if path.is_file() and path.stem not in retained_ids] if image_dir.is_dir() and label_dir.is_dir() else []
    return DatasetPlan(
        dataset_id="kitti2d-local-100",
        task_id="detection2d",
        source_root=_relative_to(data_root, source_root),
        modalities=("image_2", "label_2"),
        retained_ids=retained_ids,
        missing_pairs=missing,
        retained_assets=_asset_records(data_root, retained_paths),
        removal_assets=_removal_records(data_root, removal_paths),
        provenance={"selection": "sha256-ranked paired IDs", "split": "local-source"},
    )


def _semantic_plan(data_root: Path, count: int) -> DatasetPlan:
    source_root = _first_existing(data_root, ("datasets/kitti_semantics/training", "datasets/kitti_semantics"))
    image_dir = source_root / "image_2"
    semantic_dir = source_root / "semantic"
    image_ids, semantic_ids = _stems(image_dir), _stems(semantic_dir)
    paired = image_ids & semantic_ids
    retained_ids = _select_ids(paired, count)
    missing = tuple(sorted(image_ids ^ semantic_ids))
    selected = set(retained_ids)
    retained_paths = [path for directory in (image_dir, semantic_dir) if directory.is_dir() for path in directory.iterdir() if path.is_file() and path.stem in selected]
    related_dirs = tuple(directory for directory in (image_dir, semantic_dir, source_root / "instance", source_root / "semantic_rgb") if directory.is_dir())
    removal_paths = [path for directory in related_dirs for path in directory.iterdir() if path.is_file() and path.stem not in selected]
    return DatasetPlan(
        dataset_id="kitti-semantic-local-100",
        task_id="segmentation",
        source_root=_relative_to(data_root, source_root),
        modalities=("image_2", "semantic"),
        retained_ids=retained_ids,
        missing_pairs=missing,
        retained_assets=_asset_records(data_root, retained_paths),
        removal_assets=_removal_records(data_root, removal_paths),
        provenance={"selection": "sha256-ranked paired IDs", "split": "training"},
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"INVALID_NUSCENES_METADATA: {path}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"INVALID_NUSCENES_METADATA: expected list in {path}")
    return [item for item in parsed if isinstance(item, dict)]


def _nuscenes_plan(data_root: Path, count: int) -> DatasetPlan:
    source_root = data_root / "nuscenes"
    metadata_root = source_root / "v1.0-mini"
    scenes = _read_rows(metadata_root / "scene.json") if (metadata_root / "scene.json").is_file() else []
    samples = _read_rows(metadata_root / "sample.json") if (metadata_root / "sample.json").is_file() else []
    sample_data = _read_rows(metadata_root / "sample_data.json") if (metadata_root / "sample_data.json").is_file() else []
    samples_by_scene: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        samples_by_scene.setdefault(str(sample.get("scene_token", "")), []).append(sample)
    selected_tokens: list[str] = []
    per_scene = 10 if count == 100 else max(1, count // max(len(scenes), 1))
    for scene in sorted(scenes, key=lambda item: str(item.get("token", ""))):
        candidates = samples_by_scene.get(str(scene.get("token", "")), [])
        candidates.sort(key=lambda item: (int(item.get("timestamp", 0)), str(item.get("token", ""))))
        selected_tokens.extend(str(item["token"]) for item in candidates[:per_scene] if item.get("token"))
    selected_tokens = selected_tokens[:count]
    selected_set = set(selected_tokens)
    selected_data = [row for row in sample_data if str(row.get("sample_token", "")) in selected_set]
    all_data_paths = [source_root / str(row["filename"]) for row in sample_data if row.get("filename")]
    retained_paths = [source_root / str(row["filename"]) for row in selected_data if row.get("filename")]
    removal_paths = [source_root / str(row["filename"]) for row in sample_data if row.get("filename") and str(row.get("sample_token", "")) not in selected_set]
    missing_tokens = tuple(sorted(str(row.get("sample_token", "")) for row in selected_data if not (source_root / str(row.get("filename", ""))).is_file()))
    return DatasetPlan(
        dataset_id="nuscenes-mini-local-100",
        task_id="detection3d",
        source_root=_relative_to(data_root, source_root),
        modalities=("camera_360", "lidar", "radar", "calibration", "ego_pose", "annotations"),
        retained_ids=tuple(selected_tokens),
        missing_pairs=missing_tokens,
        limitations=("num_sweeps=1",),
        retained_assets=_asset_records(data_root, retained_paths),
        removal_assets=_removal_records(data_root, removal_paths),
        provenance={
            "version": "v1.0-mini",
            "selection": "first ten timestamp-ordered keyframes per scene",
            "retained_sample_data": len(selected_data),
            "declared_asset_records": len(all_data_paths),
        },
    )


def _kitti3d_fixture_plan(data_root: Path) -> DatasetPlan:
    fixture_root = data_root / "training"
    ids = _select_ids(_stems(fixture_root / "image_2"), 5)
    return DatasetPlan(
        dataset_id="kitti3d-fixtures",
        task_id="detection3d",
        source_root=_relative_to(data_root, fixture_root),
        modalities=("image_2", "label_2"),
        retained_ids=ids,
        limitations=("INSUFFICIENT_FOR_REQUESTED_SUBSET", "missing calib and velodyne"),
        provenance={"purpose": "fixture-only, never a PointPillars benchmark"},
    )


def build_plan(data_root: Path | str, count: int = 100) -> CurationPlan:
    if count < 1:
        raise ValueError("CURATION_COUNT_MUST_BE_POSITIVE")
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT_NOT_FOUND: {root}")
    return CurationPlan(
        data_root=str(root),
        count=count,
        datasets={
            "kitti2d": _kitti_plan(root, count),
            "kitti_semantic": _semantic_plan(root, count),
            "nuscenes3d": _nuscenes_plan(root, count),
            "kitti3d_fixtures": _kitti3d_fixture_plan(root),
        },
    )


def validate_plan(plan: CurationPlan) -> ValidationResult:
    root = Path(plan.data_root).resolve()
    missing_pairs = tuple(
        f"{name}:{sample_id}"
        for name, dataset in plan.datasets.items()
        for sample_id in dataset.missing_pairs
    )
    bad_hashes: list[str] = []
    for name, dataset in plan.datasets.items():
        for asset in dataset.retained_assets:
            path = root / str(asset["path"])
            if not path.is_file() or _sha256(path) != asset["sha256"]:
                bad_hashes.append(f"{name}:{asset['path']}")
    nuscenes = plan.datasets.get("nuscenes3d")
    dangling_tokens: tuple[str, ...] = ()
    if nuscenes and len(nuscenes.retained_ids) and len(nuscenes.retained_ids) < plan.count:
        dangling_tokens = tuple(nuscenes.retained_ids)
    return ValidationResult(missing_pairs, tuple(bad_hashes), dangling_tokens)


def write_registry(plan: CurationPlan, output: Path) -> Path:
    registry_root = output / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    plan_path = registry_root / "curation-plan.json"
    plan_path.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    for name, dataset in plan.datasets.items():
        manifest = {
            "dataset_id": dataset.dataset_id,
            "task_id": dataset.task_id,
            "source_root": dataset.source_root,
            "modalities": dataset.modalities,
            "sample_ids": dataset.retained_ids,
            "limitations": dataset.limitations,
            "provenance": dataset.provenance,
        }
        (registry_root / f"{name}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return plan_path


def _require_uuid(operation_id: str) -> str:
    try:
        return str(uuid.UUID(operation_id))
    except ValueError as exc:
        raise ValueError("CURATION_OPERATION_ID_MUST_BE_UUID") from exc


def apply_plan(plan: CurationPlan, *, output: Path, operation_id: str) -> Path:
    operation_id = _require_uuid(operation_id)
    validation = validate_plan(plan)
    if not validation.valid:
        raise ValueError(f"CURATION_PLAN_INVALID: {validation.as_dict()}")
    root, output = Path(plan.data_root).resolve(), output.resolve()
    if output != root / "curated":
        raise ValueError("CURATION_OUTPUT_MUST_BE_DATA_ROOT_CURATED")
    quarantine = output / ".quarantine" / operation_id
    files_root = quarantine / "files"
    if quarantine.exists():
        raise FileExistsError(f"CURATION_OPERATION_ALREADY_EXISTS: {operation_id}")
    moved: list[dict[str, Any]] = []
    try:
        for dataset in plan.datasets.values():
            for asset in dataset.removal_assets:
                source = root / str(asset["path"])
                destination = files_root / str(asset["path"])
                if not source.is_file() or source.stat().st_size != asset["size_bytes"]:
                    raise ValueError(f"CURATION_SOURCE_CHANGED: {source}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                moved.append(
                    {
                        **asset,
                        "sha256": _sha256(destination),
                        "quarantine_path": destination.relative_to(quarantine).as_posix(),
                    }
                )
    except Exception:
        for item in reversed(moved):
            source, destination = root / item["path"], quarantine / item["quarantine_path"]
            if destination.is_file() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
        raise
    journal = {
        "operation_id": operation_id,
        "created_at": datetime.now(UTC).isoformat(),
        "state": "QUARANTINED",
        "data_root": str(root),
        "moved_assets": moved,
    }
    (quarantine / "journal.json").write_text(json.dumps(journal, indent=2, sort_keys=True), encoding="utf-8")
    return quarantine / "journal.json"


def finalize(output: Path, operation_id: str) -> Path:
    operation_id = _require_uuid(operation_id)
    quarantine = output.resolve() / ".quarantine" / operation_id
    journal_path = quarantine / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("state") != "QUARANTINED":
        raise ValueError("CURATION_JOURNAL_NOT_QUARANTINED")
    for asset in journal.get("moved_assets", []):
        target = quarantine / str(asset["quarantine_path"])
        if not target.is_file() or _sha256(target) != asset["sha256"]:
            raise ValueError(f"CURATION_QUARANTINE_CHANGED: {target}")
    shutil.rmtree(quarantine / "files")
    journal["state"] = "FINALIZED"
    journal["finalized_at"] = datetime.now(UTC).isoformat()
    journal_path.write_text(json.dumps(journal, indent=2, sort_keys=True), encoding="utf-8")
    return journal_path


def _load_plan(path: Path) -> CurationPlan:
    return CurationPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-plan")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--operation-id")
    parser.add_argument("--finalize")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    output = Path(args.output).resolve() if args.output else root / "curated"
    if args.validate_plan:
        result = validate_plan(_load_plan(Path(args.validate_plan)))
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0 if result.valid else 2
    if args.finalize:
        print(finalize(output, args.finalize))
        return 0
    if args.verify:
        plan_path = output / "registry" / "curation-plan.json"
        result = validate_plan(_load_plan(plan_path))
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0 if result.valid else 2
    plan = build_plan(root, args.count)
    result = validate_plan(plan)
    if args.apply:
        if not args.operation_id:
            raise ValueError("CURATION_OPERATION_ID_REQUIRED_FOR_APPLY")
        plan_path = write_registry(plan, output)
        print(apply_plan(_load_plan(plan_path), output=output, operation_id=args.operation_id))
        return 0
    if args.dry_run:
        print(json.dumps({"plan": plan.as_dict(), "validation": result.as_dict()}, indent=2, sort_keys=True))
        return 0 if result.valid else 2
    print(write_registry(plan, output))
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
