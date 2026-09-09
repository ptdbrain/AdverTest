"""Shared API helpers, config resolvers, and artifact URI utilities."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from src.adapters import get_adapter
from src.api import dependencies as api_dependencies
from src.api.jobs import SqliteRunStore
from src.api.schemas import RunJobOut, RunReportOut
from src.attacks import ATTACK_CATALOG, load_attacks
from src.config import get_settings
from src.datasets import load_datasets
from src.models import list_known_versions, portable_reference_versions, scan_base_checkpoints, scan_model_artifacts
from src.models.families import FAMILIES, adapter_request, family_for_version
from src.models.versions import ModelVersion
from src.pipeline import RunConfig


def validate_uploaded_image(payload: bytes) -> tuple[str, tuple[int, int]]:
    """Validate image bytes and return (format, (width, height))."""
    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            return image.format or "unknown", image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_IMAGE",
                "message": "Payload is not a decodable image.",
            },
        ) from exc


def upload_root() -> Path:
    """Return configured upload directory."""
    root = Path(get_settings().data_root).expanduser().resolve() / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def dataset_root() -> Path:
    """Return configured datasets directory."""
    root = Path(get_settings().data_root).expanduser().resolve() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_uri(path_value: Any) -> str | None:
    """Convert absolute filesystem path to browser-readable static data URI."""
    if not path_value:
        return None
    raw_value = str(path_value)
    if raw_value.startswith(("https://", "http://")):
        return raw_value
    candidate = Path(raw_value).expanduser().resolve()
    settings = get_settings()
    artifact_root = Path(settings.artifact_root).expanduser().resolve()
    if candidate == artifact_root or artifact_root in candidate.parents:
        return f"/data/artifacts/{candidate.relative_to(artifact_root).as_posix()}"
    data_root = Path(settings.data_root).expanduser().resolve()
    if candidate != data_root and data_root not in candidate.parents:
        return None
    return f"/data/{candidate.relative_to(data_root).as_posix()}"


def sample_with_artifact_urls(sample: dict[str, Any]) -> dict[str, Any]:
    """Expose browser-readable artifact URIs without leaking local filesystem paths."""
    result = {key: value for key, value in sample.items() if not key.endswith("_path")}
    existing_artifacts = sample.get("artifacts")
    artifacts = {
        key: _browser_artifact_value(value)
        for key, value in (existing_artifacts.items() if isinstance(existing_artifacts, dict) else ())
    }
    artifacts.update({
        "clean_input_url": artifact_uri(sample.get("clean_image_path")),
        "attacked_input_url": artifact_uri(sample.get("attacked_image_path")),
        "clean_prediction_url": artifact_uri(sample.get("clean_prediction_path")),
        "attacked_prediction_url": artifact_uri(sample.get("attacked_prediction_path")),
    })
    result["artifacts"] = artifacts
    return result


def _browser_artifact_value(value: Any) -> str | None:
    if not value:
        return None
    raw = str(value)
    if raw.startswith(("https://", "http://", "data:", "blob:", "/data/")):
        return raw
    return artifact_uri(raw)


def registered_model_versions(store: SqliteRunStore | None = None) -> list[ModelVersion]:
    """Keep every planned lineage role visible even when its artifact is absent."""
    effective_store = store or api_dependencies.get_store()
    placeholders = list_known_versions()
    by_role = {str(item.training_metadata.get("role", "")): item for item in placeholders}
    versions = {item.id: item for item in placeholders}
    settings = get_settings()
    portable_enabled = getattr(settings, "bootstrap_portable_demo", False)

    for reference in portable_reference_versions(enabled=portable_enabled):
        versions[reference.id] = reference

    if not portable_enabled:
        for base in scan_base_checkpoints(Path(settings.checkpoint_root)):
            versions[base.id] = base
        for discovered in scan_model_artifacts(Path(settings.runs_root)):
            placeholder = by_role.get(str(discovered.training_metadata.get("role", "")))
            if placeholder:
                discovered = replace(
                    discovered,
                    id=placeholder.id,
                    parent_id=placeholder.parent_id,
                    parent_lineage=placeholder.parent_lineage,
                    model_family_id=placeholder.model_family_id,
                    checkpoint_role=placeholder.checkpoint_role,
                )
            versions[discovered.id] = discovered
    for record in effective_store.list_records("checkpoint"):
        if record.get("status") != "READY":
            continue
        task_id = str(record.get("task_id", ""))
        family_id = str(record.get("family_id", ""))
        path = Path(str(record.get("storage_path", ""))).expanduser().resolve()
        family = FAMILIES.get(family_id)
        if family is None or task_id not in family.supported_tasks or not path.is_file():
            continue
        role = str(record.get("checkpoint_role", "base"))
        is_checkpoint_required = family.requires_checkpoint
        is_runnable = family.runnable and not (portable_enabled and is_checkpoint_required)
        blocked_reason = (
            "PORTABLE_REFERENCE_ONLY"
            if (portable_enabled and is_checkpoint_required)
            else family.blocked_reason
        )
        versions[str(record["checkpoint_id"])] = ModelVersion(
            id=str(record["checkpoint_id"]),
            model_name=str(record.get("display_name", record["checkpoint_id"])),
            task=task_id,  # type: ignore[arg-type]
            checkpoint_path=str(path),
            checkpoint_hash=str(record.get("sha256")),
            parent_id=record.get("parent_checkpoint_id"),
            training_metadata={
                "source": "user_upload",
                "model_config": record.get("model_config"),
                "validation": record.get("validation", {}),
                "training_dataset_version_id": record.get("training_dataset_version_id"),
            },
            runnable=is_runnable,
            blocked_reason=blocked_reason,
            checkpoint_validated=True,
            gate_outcome="PASSED",
            evidence_tier="QUARANTINE_VALIDATED",
            model_family_id=family_id,
            checkpoint_role=role,  # type: ignore[arg-type]
        )
    return sorted(versions.values(), key=lambda item: item.id)


def _native_class_names(checkpoint_id: str) -> tuple[str, ...] | None:
    """Checkpoint's native output classes, or None when the platform DB does not know it.

    Locally scanned checkpoints (demo flows) have no platform record, so the
    target check is skipped for them rather than blocking every legacy run.
    """
    try:
        from src.api.platform_dependencies import get_platform_database
        from src.persistence.models import CheckpointRecord

        with get_platform_database().session() as db:
            record = db.get(CheckpointRecord, checkpoint_id)
            if record is None:
                return None
            return tuple(json.loads(record.native_class_names_json))
    except Exception:  # pragma: no cover - platform DB unavailable in legacy setups
        return None


def _project_checkpoint_version(project_id: str, checkpoint_id: str) -> ModelVersion | None:
    """Resolve one validated platform checkpoint without exposing other projects."""
    try:
        from sqlalchemy import select

        from src.api.platform_dependencies import get_platform_artifacts, get_platform_database
        from src.core.platform_contracts import ArtifactState, CheckpointStatus
        from src.persistence.models import ArtifactRecord, CheckpointRecord

        with get_platform_database().session() as db:
            row = db.execute(
                select(CheckpointRecord, ArtifactRecord)
                .join(ArtifactRecord, ArtifactRecord.id == CheckpointRecord.artifact_id)
                .where(CheckpointRecord.id == checkpoint_id, CheckpointRecord.project_id == project_id)
            ).first()
            if row is None:
                return None
            checkpoint, artifact = row
            family = FAMILIES.get(checkpoint.model_family_id)
            portable_mode_enabled = getattr(get_settings(), "bootstrap_portable_demo", False)
            is_checkpoint_required = family.requires_checkpoint if family else True
            runnable = bool(
                checkpoint.status == CheckpointStatus.READY.value
                and artifact.state == ArtifactState.READY.value
                and family
                and family.runnable
                and not (portable_mode_enabled and is_checkpoint_required)
            )
            if checkpoint.status != CheckpointStatus.READY.value:
                blocked_reason = checkpoint.validation_error_code or checkpoint.status
            elif artifact.state != ArtifactState.READY.value:
                blocked_reason = artifact.state
            elif portable_mode_enabled and is_checkpoint_required:
                blocked_reason = "PORTABLE_REFERENCE_ONLY"
            else:
                blocked_reason = family.blocked_reason if family else "MODEL_FAMILY_UNKNOWN"
            filename = Path(artifact.original_filename).name or f"{checkpoint.id}.pt"
            artifact_id = artifact.id
            sha256 = artifact.sha256
            native_class_names = tuple(json.loads(checkpoint.native_class_names_json))
            model_family_id = checkpoint.model_family_id
            task_id = checkpoint.task_id
        checkpoint_root = (
            Path(get_settings().data_root).expanduser().resolve()
            / "project-assets"
            / project_id
            / "checkpoints"
            / checkpoint_id
        )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        path = checkpoint_root / filename
        if runnable and not path.is_file():
            path.write_bytes(get_platform_artifacts().read_bytes(project_id, artifact_id))
        return ModelVersion(
            id=checkpoint_id,
            model_name=filename,
            task=task_id,  # type: ignore[arg-type]
            checkpoint_path=str(path) if runnable else None,
            checkpoint_hash=sha256,
            parent_id=None,
            training_metadata={"source": "project_upload", "native_class_names": native_class_names},
            runnable=runnable,
            blocked_reason=None if runnable else blocked_reason,
            checkpoint_validated=runnable,
            gate_outcome="PASSED" if runnable else "WAITING",
            evidence_tier="QUARANTINE_VALIDATED" if runnable else "UNVERIFIED",
            model_family_id=model_family_id,
            checkpoint_role="base",
        )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        return None


def _validate_class_mapping(config: RunConfig, checkpoint_id: str | None) -> None:
    """Workflow §4 gate: reject runs whose label mapping is incomplete or wrong.

    Only a supplied mapping is enforced — legacy runs without one keep working.
    """
    required = set(config.dataset_class_names)
    if required and not config.class_mapping:
        raise HTTPException(
            status_code=409,
            detail={"code": "CLASS_MAPPING_REQUIRED", "unmapped_classes": sorted(required)},
        )
    if not config.class_mapping:
        return
    unmapped = sorted(
        required.difference(config.class_mapping)
        | {name for name, target in config.class_mapping.items() if target is None or target == ""}
    )
    if unmapped:
        raise HTTPException(
            status_code=409,
            detail={"code": "CLASS_MAPPING_INCOMPLETE", "unmapped_classes": unmapped},
        )
    if checkpoint_id is None:
        return
    native = _native_class_names(checkpoint_id)
    if native is None:
        return
    known = set(native)
    unknown = sorted(
        {target for target in config.class_mapping.values() if target and target != "ignore" and target not in known}
    )
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "CLASS_MAPPING_TARGET_UNKNOWN",
                "unknown_classes": unknown,
                "native_class_names": sorted(known),
            },
        )


def _resolve_project_dataset(config: RunConfig) -> RunConfig:
    if not config.dataset_version_id:
        return config
    if not config.project_id:
        raise HTTPException(status_code=409, detail={"code": "PROJECT_ASSET_SCOPE_REQUIRED"})
    try:
        from sqlalchemy import select

        from src.api.platform_dependencies import get_platform_artifacts, get_platform_database
        from src.core.platform_contracts import ArtifactState
        from src.persistence.models import ArtifactRecord, DatasetVersionRecord

        with get_platform_database().session() as db:
            row = db.execute(
                select(DatasetVersionRecord, ArtifactRecord)
                .join(ArtifactRecord, ArtifactRecord.id == DatasetVersionRecord.artifact_id)
                .where(
                    DatasetVersionRecord.id == config.dataset_version_id,
                    DatasetVersionRecord.project_id == config.project_id,
                )
            ).first()
            if row is None:
                raise HTTPException(
                    status_code=404,
                    detail={"code": "DATASET_VERSION_UNKNOWN", "dataset_version_id": config.dataset_version_id},
                )
            version, artifact = row
            manifest = json.loads(version.manifest_json)
            runtime = manifest.get("_advertest_runtime", {}) if isinstance(manifest, dict) else {}
            if version.status != "READY" or artifact.state != ArtifactState.READY.value or not runtime.get("runnable"):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "code": "DATASET_NOT_RUNNABLE",
                        "dataset_version_id": version.id,
                        "reason": runtime.get("blocked_reason") or version.status,
                    },
                )
            artifact_id = artifact.id
            dataset_name = str(runtime["dataset"])
            dataset_params = dict(runtime.get("dataset_params", {}))
            class_map = manifest.get("class_map", {}) if isinstance(manifest, dict) else {}
            dataset_class_names = [str(name) for name in class_map.values()] if isinstance(class_map, dict) else []
        target = (
            Path(get_settings().data_root).expanduser().resolve()
            / "project-assets"
            / config.project_id
            / "datasets"
            / config.dataset_version_id
        )
        marker = target / ".materialized"
        if not marker.is_file():
            target.mkdir(parents=True, exist_ok=True)
            content = get_platform_artifacts().read_bytes(config.project_id, artifact_id)
            with zipfile.ZipFile(BytesIO(content)) as archive:
                members = archive.namelist()
                if any(name.startswith("/") or ".." in name.split("/") for name in members):
                    raise HTTPException(status_code=422, detail={"code": "DATASET_ARCHIVE_INVALID"})
                archive.extractall(target)
            marker.touch()
        dataset_params["root"] = str(target)
        return config.model_copy(
            update={
                "dataset": dataset_name,
                "dataset_params": dataset_params,
                "dataset_class_names": dataset_class_names,
            }
        )
    except HTTPException:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail={"code": "DATASET_MATERIALIZATION_FAILED"}) from exc
def resolve_run_config(
    config: RunConfig,
    *,
    allow_defence: bool = False,
    store: SqliteRunStore | None = None,
) -> RunConfig:
    """Resolve a product ModelVersion server-side; browsers never receive weight paths."""
    config = _resolve_project_dataset(config)
    checkpoint_id = config.checkpoint_id or config.model_version_id
    if not checkpoint_id:
        return config
    versions = registered_model_versions(store)
    version = next((item for item in versions if item.id == checkpoint_id), None)
    if version is None and config.project_id:
        version = _project_checkpoint_version(config.project_id, checkpoint_id)
    if version is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "CHECKPOINT_UNKNOWN", "checkpoint_id": checkpoint_id},
        )
    if config.model_family_id is not None and config.model_family_id != version.model_family_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "CHECKPOINT_FAMILY_MISMATCH",
                "model_family_id": config.model_family_id,
                "checkpoint_id": checkpoint_id,
            },
        )
    if config.task_id is not None and config.task_id != version.task:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "MODEL_FAMILY_TASK_MISMATCH",
                "task_id": config.task_id,
                "model_version_id": config.model_version_id,
                "model_task": version.task,
            },
        )
    if version.checkpoint_role != "base" and not allow_defence:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "DEFENCE_CHECKPOINT_NOT_ALLOWED_IN_ATTACK",
                "checkpoint_id": checkpoint_id,
                "checkpoint_role": version.checkpoint_role,
            },
        )
    try:
        dataset_cls = load_datasets().get(config.dataset)
    except KeyError:
        dataset_cls = None
    if dataset_cls is not None and config.task_id is not None and dataset_cls.task_id != config.task_id:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "TASK_DATASET_MISMATCH",
                "task_id": config.task_id,
                "dataset": config.dataset,
                "dataset_task_id": dataset_cls.task_id,
            },
        )
    _validate_class_mapping(config, checkpoint_id)
    attack_names = list(config.attacks)
    if config.recipe:
        attack_names.extend(step.attack_name for step in config.recipe.steps)
    for attack_name in dict.fromkeys(attack_names):
        load_attacks().get(attack_name)
        allowed = ATTACK_CATALOG.get(attack_name).task_ids
        if version.task not in allowed:
            raise HTTPException(
                status_code=422,
                detail={"code": "ATTACK_NOT_COMPATIBLE", "attack": attack_name, "task_id": version.task},
            )
    if not version.runnable:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MODEL_NOT_RUNNABLE",
                "model_version_id": config.model_version_id,
                "reason": version.blocked_reason,
            },
        )
    checkpoint = ""
    if version.checkpoint_path:
        path = Path(version.checkpoint_path).resolve()
        if not path.is_file():
            raise HTTPException(
                status_code=409,
                detail={"code": "CHECKPOINT_MISSING", "model_version_id": config.model_version_id},
            )
        checkpoint = str(path)
    elif family_for_version(version).requires_checkpoint:
        if not version.checkpoint_path:
            raise HTTPException(
                status_code=409,
                detail={"code": "MODEL_NOT_RUNNABLE", "model_version_id": config.model_version_id, "reason": version.blocked_reason},
            )
    settings = get_settings()
    try:
        adapter_name, adapter_params = adapter_request(
            version, checkpoint=checkpoint, config=config, settings=settings
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": str(exc), "model_version_id": config.model_version_id},
        ) from exc
    return config.model_copy(
        update={
            "model": adapter_name,
            "adapter_params": adapter_params,
            "task_id": version.task,
            "model_family_id": version.model_family_id,
            "checkpoint_id": version.id,
        }
    )


def attack_catalog_availability(
    *,
    task_id: str | None,
    model_family_id: str | None,
    checkpoint_id: str | None,
    dataset_name: str | None,
    store: SqliteRunStore | None = None,
) -> tuple[str | None, dict[str, tuple[str, ...]]] | None:
    """Evaluate attack cards against the selected product contracts."""
    if model_family_id is None and checkpoint_id is None:
        return None

    if checkpoint_id:
        versions = registered_model_versions(store)
        version = next((item for item in versions if item.id == checkpoint_id), None)
        if version is None:
            return "CHECKPOINT_UNKNOWN", {}
        if model_family_id is not None and version.model_family_id != model_family_id:
            return "CHECKPOINT_FAMILY_MISMATCH", {}
        if task_id is not None and version.task != task_id:
            return "MODEL_FAMILY_TASK_MISMATCH", {}
        if version.checkpoint_role != "base":
            return "DEFENCE_CHECKPOINT_NOT_ALLOWED_IN_ATTACK", {}
        if not version.runnable:
            return f"MODEL_NOT_RUNNABLE:{version.blocked_reason or 'CHECKPOINT_MISSING'}", {}
        try:
            checkpoint = ""
            if version.checkpoint_path:
                path = Path(version.checkpoint_path).resolve()
                if not path.is_file():
                    return "CHECKPOINT_MISSING", {}
                checkpoint = str(path)
            elif family_for_version(version).requires_checkpoint:
                return f"MODEL_NOT_RUNNABLE:{version.blocked_reason or 'CHECKPOINT_MISSING'}", {}
            adapter_name, adapter_params = adapter_request(
                version, checkpoint=checkpoint, config=RunConfig(), settings=get_settings()
            )
            info = get_adapter(adapter_name, **adapter_params).metadata()
        except ValueError as exc:
            return str(exc), {}
        effective_task = info.task
        capabilities = info.capabilities
    else:
        family = FAMILIES.get(model_family_id or "")
        if family is None:
            return "MODEL_FAMILY_UNKNOWN", {}
        if task_id is not None and task_id not in family.supported_tasks:
            return "MODEL_FAMILY_TASK_MISMATCH", {}
        if not family.runnable:
            return f"MODEL_FAMILY_NOT_RUNNABLE:{family.blocked_reason or family.id}", {}
        return "BASE_CHECKPOINT_REQUIRED", {}

    if dataset_name is None:
        return "DATASET_REQUIRED", {}
    try:
        dataset_cls = load_datasets().get(dataset_name)
    except KeyError:
        return "DATASET_UNKNOWN", {}
    if dataset_cls.task_id != effective_task:
        return "TASK_DATASET_MISMATCH", {}

    annotation_types = frozenset(
        annotation
        for annotation in (
            "boxes" if any(value.startswith("boxes") for value in dataset_cls.annotation_schema) else None,
            "mask" if any("mask" in value for value in dataset_cls.annotation_schema) else None,
        )
        if annotation is not None
    )
    result = ATTACK_CATALOG.list(
        task=effective_task,
        model_capabilities=capabilities,
        annotation_types=annotation_types,
        modality=dataset_cls.modality,
        online=False,
    )
    exclusions = {item.name: item.reasons for item in result.exclusions}
    return None, exclusions


def comparison_signature(report: dict[str, Any]) -> dict[str, Any]:
    """Identity needed for a paired recovery claim; model identity is deliberately excluded."""
    provenance = report.get("provenance") or {}
    run_config = provenance.get("run_config") or {}
    samples = report.get("sample_results") or []
    return {
        "dataset_version_id": provenance.get("dataset_version_id")
        or run_config.get("dataset_version_id")
        or report.get("dataset"),
        "benchmark_protocol_id": provenance.get("benchmark_protocol_id") or run_config.get("benchmark_protocol_id"),
        "recipe_hash": (provenance.get("recipe") or {}).get("recipe_hash"),
        "sample_ids": sorted(str(item.get("sample_id")) for item in samples if item.get("sample_id")),
        "seed": run_config.get("seed"),
        "limit": run_config.get("limit"),
        "iou_threshold": run_config.get("iou_threshold"),
        "confidence_threshold": run_config.get("confidence_threshold"),
        "preprocessing": run_config.get("preprocessing_version"),
        "image_size": run_config.get("image_size"),
    }


def comparison_metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    """Keep base/candidate metric families explicit rather than one blended score."""
    output: dict[str, dict[str, float | str]] = {
        "clean_detection_score": {
            "value": float(right.get("ap_clean", 0.0)) - float(left.get("ap_clean", 0.0)),
            "unit": "ratio",
        },
    }
    for input_name, payload in (("clean", left), ("attacked", right)):
        if input_name == "attacked":
            continue
        for metric in ("ap50", "ap75", "map50_95"):
            before = _metric_value(payload, metric, attacked=False)
            after = _metric_value(right, metric, attacked=False)
            output[f"clean_{metric}"] = {"value": after - before, "unit": "ratio"}
    for metric in ("ap50", "ap75", "map50_95"):
        before = _metric_value(left, metric, attacked=True)
        after = _metric_value(right, metric, attacked=True)
        output[f"attacked_{metric}"] = {"value": after - before, "unit": "ratio"}
    return output


def _metric_value(report: dict[str, Any], metric: str, *, attacked: bool) -> float:
    if not attacked:
        values = (report.get("metrics") or {}).get("clean") or {}
        if metric == "ap50":
            return float(values.get(metric, report.get("ap_clean", 0.0)))
        return float(values.get(metric, report.get("ap_clean", 0.0)))
    cells = report.get("cells") or []
    if not cells:
        return 0.0
    values = cells[-1].get("metrics") or {}
    return float(values.get(metric, cells[-1].get("ap", 0.0)))


def mean_attack_score(report: dict[str, Any]) -> float:
    """Compute average AP across evaluated attack cells."""
    cells = report.get("cells", [])
    if not cells:
        return float(report.get("ap_clean", 0.0))
    return sum(float(cell["ap"]) for cell in cells) / len(cells)


def is_failure_case(payload: dict[str, Any]) -> bool:
    """Accept only benchmark samples carrying an explicit failure signal."""
    degradation = payload.get("degradation_hint")
    has_positive_degradation = (
        isinstance(degradation, (int, float)) and not isinstance(degradation, bool) and degradation > 0.0
    )
    has_reason = any(
        isinstance(payload.get(field), str) and bool(payload[field].strip()) for field in ("reason", "failure_reason")
    )
    return has_positive_degradation or has_reason or payload.get("failed") is True


def require_run(store: SqliteRunStore, run_id: str) -> dict[str, Any]:
    """Retrieve run record or raise 404."""
    item = store.get(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return item


def require_completed_report(store: SqliteRunStore, run_id: str) -> dict[str, Any]:
    """Retrieve run report or raise 404/409."""
    item = require_run(store, run_id)
    if item.get("report") is None:
        raise HTTPException(status_code=409, detail=f"run {run_id!r} has no completed report")
    return item["report"]


def job_out(item: dict[str, Any] | None) -> RunJobOut:
    """Convert raw run dictionary to RunJobOut schema."""
    if item is None:
        raise HTTPException(status_code=404, detail="unknown run")
    report = RunReportOut(**item["report"]) if item.get("report") else None
    return RunJobOut(
        run_id=item["run_id"],
        status=item["status"],
        progress=item["progress"],
        detail=item["detail"],
        report=report,
        error=item.get("error"),
    )


# Compatibility aliases
_validate_uploaded_image = validate_uploaded_image
_upload_root = upload_root
_dataset_root = dataset_root
_artifact_uri = artifact_uri
_sample_with_artifact_urls = sample_with_artifact_urls
_registered_model_versions = registered_model_versions
_resolve_run_config = resolve_run_config
_attack_catalog_availability = attack_catalog_availability
_comparison_signature = comparison_signature
_comparison_metric_deltas = comparison_metric_deltas
_mean_attack_score = mean_attack_score
_is_failure_case = is_failure_case
_require_run = require_run
_require_completed_report = require_completed_report
_job_out = job_out
