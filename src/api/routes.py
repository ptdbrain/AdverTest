"""AdverTest API: catalog plus durable, asynchronous test-run jobs."""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect

from src.adapters import load_adapters
from src.api.generated_dataset_service import GeneratedDatasetService
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.schemas import (
    AttackCatalogItem,
    ClosedLoopAdvanceIn,
    ClosedLoopSnapshotOut,
    ClosedLoopStartIn,
    CostEstimateOut,
    CreateReviewIn,
    DatasetCatalogItem,
    DatasetImportIn,
    EvidenceStatusOut,
    FailureClusterCreateIn,
    GeneratedDatasetCreateIn,
    GeneratedDatasetEventsOut,
    GeneratedDatasetJobOut,
    GeneratedDatasetManifestOut,
    GeneratedDatasetValidationOut,
    GeneratedDatasetVariantsOut,
    LineageGraphOut,
    ModelCatalogItem,
    ModelComparisonIn,
    ModelVersionOut,
    PerceptionModeOut,
    PreflightOut,
    RecipePreviewIn,
    RecipeRandomizeIn,
    RecipeRecordIn,
    RecipeSweepIn,
    RecipeValidationIn,
    RecipeValidationOut,
    ResolveReviewIn,
    RetrainingBacklogIn,
    RetrainingBacklogItemIn,
    ReviewOut,
    RunJobOut,
    RunReportOut,
    TrainingRunIn,
)
from src.api.training_service import TrainingJobService
from src.api.workflow_store import WorkflowJobStore
from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.recipes import RecipeBuilder
from src.config import get_settings
from src.core.hashing import stable_digest
from src.datasets import load_datasets
from src.datasets.folder import FolderDataset
from src.datasets.versioning import DatasetIngestor, IngestConfig
from src.evaluation.export import export_comparison
from src.models import list_known_versions, scan_model_artifacts
from src.pipeline import RunConfig, TestRunner
from src.services.person_d import PersonDServices
from src.training.contracts import DefenseProfile, TrainingRunConfig

router = APIRouter()
_runner = TestRunner()
_store = SqliteRunStore(get_settings().database_url)
_worker = LocalRunWorker(_store, max_workers=get_settings().worker_max_concurrency)
_workflow_store = WorkflowJobStore(get_settings().database_url)
_training_jobs = TrainingJobService(
    _workflow_store,
    PersonDServices.default().training.registry,
    max_workers=get_settings().worker_max_concurrency,
)
_generated_datasets = GeneratedDatasetService(
    _workflow_store,
    _store,
    get_settings().artifact_root,
    max_workers=get_settings().worker_max_concurrency,
)
_generated_datasets.recover()
for _run_id, _config in _store.recoverable():
    _worker.enqueue(_run_id, _config)
_training_jobs.recover()


@router.post("/uploads/images", status_code=201)
async def upload_image(request: Request) -> dict[str, Any]:
    """Store one user-supplied image; clients may send raw bytes, no multipart needed."""
    filename = request.headers.get("x-filename", "upload.bin")
    filename = Path(filename).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename):
        raise HTTPException(status_code=422, detail="invalid x-filename")
    payload = await request.body()
    if not payload or len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="image payload must be between 1 byte and 25 MiB")
    target_dir = _store.path.parent / "uploads"
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    stored = images_dir / f"{uuid.uuid4().hex[:12]}-{filename}"
    stored.write_bytes(payload)
    manifest = target_dir / "dataset.json"
    if not manifest.is_file():
        manifest.write_text('{"anonymized": true, "split": "upload"}', encoding="utf-8")
    return {
        "upload_id": stored.stem,
        "filename": filename,
        "path": str(stored),
        "bytes": len(payload),
        "dataset": "folder_dataset",
        "dataset_params": {
            "root": str(target_dir),
            "input_format": "advertest",
        },
    }


@router.post("/datasets/import", status_code=201)
async def import_dataset(body: DatasetImportIn) -> dict[str, Any]:
    """Version an explicitly supplied, annotated local folder without copying it."""
    root = Path(body.root).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(status_code=422, detail={"code": "DATASET_ROOT_MISSING", "message": "dataset root does not exist"})
    source = FolderDataset(
        root=str(root), input_format=body.input_format,
        anonymization_manifest=body.anonymization_manifest, max_samples=body.max_samples,
    )
    source.require_anonymized()
    version = DatasetIngestor(_store.path.parent / "dataset-versions").ingest(
        source, IngestConfig(name=body.name, logical_source_id=body.logical_source_id,
                             metadata={"input_format": body.input_format}),
    )
    payload = version.model_dump(mode="json")
    payload["generation_source"] = {
        "input_dir": str(root),
        "input_format": body.input_format,
        "anonymization_manifest": body.anonymization_manifest,
    }
    return _store.put_record("dataset_version", version.version_id, payload)


@router.post("/defense-profiles", status_code=201)
async def create_defense_profile(body: DefenseProfile) -> dict[str, Any]:
    return _store.put_record("defense_profile", body.profile_id, body.model_dump(mode="json"))


@router.get("/defense-profiles/{profile_id}")
async def get_defense_profile(profile_id: str) -> dict[str, Any]:
    profile = _store.get_record("defense_profile", profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"unknown defense profile {profile_id!r}")
    return profile


@router.post("/training-runs/estimate")
async def estimate_training_run(body: TrainingRunIn) -> dict[str, Any]:
    """Estimate a registered trainer without scheduling external training."""
    config = TrainingRunConfig(run_id=f"estimate-{uuid.uuid4().hex}", **body.model_dump(mode="json"))
    try:
        return _training_jobs.estimate(config)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"TRAINER_NOT_AVAILABLE: {exc}") from exc


@router.post("/training-runs", status_code=202)
async def start_training_run(body: TrainingRunIn) -> dict[str, Any]:
    """Queue training only after the requested parent checkpoint is runnable."""
    version = next(
        (item for item in scan_model_artifacts(Path(get_settings().runs_root)) if item.id == body.model_version),
        None,
    )
    if version is None or not version.runnable:
        raise HTTPException(status_code=409, detail="WAITING_FOR_ARTIFACTS")
    config = TrainingRunConfig(run_id=f"queued-{uuid.uuid4().hex}", **body.model_dump(mode="json"))
    try:
        job_id = _training_jobs.enqueue(config)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"TRAINER_NOT_AVAILABLE: {exc}") from exc
    return _training_jobs.get(job_id) or {"id": job_id, "status": "QUEUED"}


@router.get("/training-runs")
async def list_training_runs() -> list[dict[str, Any]]:
    return _training_jobs.list()


@router.get("/training-runs/{job_id}")
async def get_training_run(job_id: str) -> dict[str, Any]:
    job = _training_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="TRAINING_RUN_UNKNOWN")
    return job


@router.post("/training-runs/{job_id}/cancel")
async def cancel_training_run(job_id: str) -> dict[str, Any]:
    if _training_jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="TRAINING_RUN_UNKNOWN")
    _training_jobs.cancel(job_id)
    return _training_jobs.get(job_id) or {"id": job_id, "status": "CANCEL_REQUESTED"}


@router.get("/training-runs/{job_id}/checkpoints")
async def get_training_checkpoints(job_id: str) -> list[dict[str, Any]]:
    if _training_jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="TRAINING_RUN_UNKNOWN")
    return _workflow_store.checkpoints(job_id)


@router.get("/training-runs/{job_id}/events")
async def get_training_events(job_id: str) -> list[dict[str, Any]]:
    if _training_jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="TRAINING_RUN_UNKNOWN")
    return _workflow_store.events(job_id)


@router.get("/training-dataset-manifests/{manifest_id}")
async def get_training_dataset_manifest(manifest_id: str) -> dict[str, Any]:
    """Retrieve a training dataset manifest record with lineage details."""
    manifest = _store.get_record("training_dataset_manifest", manifest_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"unknown training dataset manifest {manifest_id!r}")
    return manifest


@router.websocket("/training-runs/{job_id}/events/ws")
async def training_run_events(job_id: str, websocket: WebSocket) -> None:
    """WebSocket stream for background training job events."""
    await websocket.accept()
    if _training_jobs.get(job_id) is None:
        await websocket.send_json({"error": "unknown training run"})
        await websocket.close(code=4404)
        return
    cursor = 0
    try:
        while True:
            events = _workflow_store.events(job_id)
            for event in events[cursor:]:
                cursor += 1
                await websocket.send_json(event)
            job = _training_jobs.get(job_id)
            if job and job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return



@router.post("/retraining-backlogs", status_code=201)
async def create_retraining_backlog(body: RetrainingBacklogIn) -> dict[str, Any]:
    return _workflow_store.create_backlog(body.name)


@router.get("/retraining-backlogs/{backlog_id}")
async def get_retraining_backlog(backlog_id: str) -> dict[str, Any]:
    backlog = _workflow_store.get_backlog(backlog_id)
    if backlog is None:
        raise HTTPException(status_code=404, detail="BACKLOG_UNKNOWN")
    return backlog


@router.post("/retraining-backlogs/{backlog_id}/items", status_code=201)
async def add_retraining_backlog_item(backlog_id: str, body: RetrainingBacklogItemIn) -> dict[str, Any]:
    try:
        return _workflow_store.add_backlog_item(backlog_id, body.failure_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"code": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc


@router.post("/retraining-backlogs/{backlog_id}/approve")
async def approve_retraining_backlog(backlog_id: str) -> dict[str, Any]:
    try:
        return _workflow_store.approve_backlog(backlog_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"code": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc


@router.post("/model-comparisons", status_code=201)
async def create_model_comparison(body: ModelComparisonIn) -> dict[str, Any]:
    baseline, candidate = _require_run(body.baseline_run_id), _require_run(body.candidate_run_id)
    if baseline["report"] is None or candidate["report"] is None:
        raise HTTPException(status_code=409, detail="both runs must complete before comparison")
    left, right = baseline["report"], candidate["report"]
    paired = left["dataset"] == right["dataset"] and left["n_samples"] == right["n_samples"]
    comparison_id = f"comparison-{stable_digest(body.model_dump(mode='json'), length=20)}"
    metric_deltas = (
        {"clean_detection_score": {"value": right["ap_clean"] - left["ap_clean"], "unit": "ratio"}}
        if paired
        else {}
    )
    baseline_attack_score = _mean_attack_score(left)
    candidate_attack_score = _mean_attack_score(right)
    lost_score = left["ap_clean"] - baseline_attack_score
    recovered_score = candidate_attack_score - baseline_attack_score
    recovery_ratio = None if lost_score <= 0 else recovered_score / lost_score
    payload = {
        "comparison_id": comparison_id, **body.model_dump(mode="json"), "paired": paired,
        "incompatibilities": [] if paired else ["dataset_or_sample_count"],
        "metric_deltas": metric_deltas,
        "recovery_report": {
            "baseline_clean": left["ap_clean"],
            "candidate_clean": right["ap_clean"],
            "recovery_rate": {
                "ratio_value": recovery_ratio,
                "percent_value": None if recovery_ratio is None else recovery_ratio * 100,
                "unit": "percent",
            },
        },
    }
    return _store.put_record("model_comparison", comparison_id, payload)


@router.get("/model-comparisons/{comparison_id}")
async def get_model_comparison(comparison_id: str) -> dict[str, Any]:
    record = _store.get_record("model_comparison", comparison_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown comparison {comparison_id!r}")
    return record


@router.get("/model-comparisons/{comparison_id}/export")
async def export_model_comparison(comparison_id: str, format: str = Query(default="json")) -> Response:
    comparison = await get_model_comparison(comparison_id)
    try:
        artifact = export_comparison(comparison, format)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        artifact.content,
        media_type=artifact.media_type,
        headers={"Content-Disposition": f'attachment; filename="{artifact.filename}"', "X-Content-SHA256": artifact.sha256},
    )


@router.get("/model-comparisons/{comparison_id}/metric-deltas")
async def get_model_comparison_metric_deltas(comparison_id: str) -> dict[str, Any]:
    return (await get_model_comparison(comparison_id)).get("metric_deltas", {})


@router.get("/model-comparisons/{comparison_id}/recovery-report")
async def get_model_comparison_recovery_report(comparison_id: str) -> dict[str, Any]:
    return (await get_model_comparison(comparison_id)).get("recovery_report", {})


@router.get("/model-comparisons/{comparison_id}/failures")
async def get_model_comparison_failures(comparison_id: str) -> dict[str, Any]:
    """Return failure deltas between baseline and candidate runs in a comparison."""
    comp = await get_model_comparison(comparison_id)
    baseline_id = comp.get("baseline_run_id")
    candidate_id = comp.get("candidate_run_id")
    base_run = _store.get(baseline_id) if baseline_id else None
    cand_run = _store.get(candidate_id) if candidate_id else None
    base_failures = (base_run.get("report") or {}).get("worst_cases", []) if base_run else []
    cand_failures = (cand_run.get("report") or {}).get("worst_cases", []) if cand_run else []
    return {
        "comparison_id": comparison_id,
        "baseline_failures": base_failures,
        "candidate_failures": cand_failures,
        "recovered_count": max(0, len(base_failures) - len(cand_failures)),
    }



@router.post("/attack-recipes", status_code=201)
async def create_recipe(body: RecipeRecordIn) -> dict[str, Any]:
    return _store.put_record("attack_recipe", body.id, body.model_dump(mode="json"))


@router.get("/attack-recipes/{recipe_id}")
async def get_recipe(recipe_id: str) -> dict[str, Any]:
    recipe = _store.get_record("attack_recipe", recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"unknown recipe {recipe_id!r}")
    return recipe


@router.get("/catalog/recipes/presets")
async def list_recipe_presets() -> list[dict[str, Any]]:
    """Return standard pre-configured attack recipe presets."""
    return [
        {
            "preset_id": "weather_robustness",
            "name": "Weather Robustness Suite",
            "task": "detection2d",
            "steps": [
                {"attack": "fog", "severity": 2},
                {"attack": "snow", "severity": 2},
                {"attack": "frost", "severity": 3},
            ],
        },
        {
            "preset_id": "sensor_fault_suite",
            "name": "Sensor Fault & Noise Suite",
            "task": "detection2d",
            "steps": [
                {"attack": "gaussian_noise", "severity": 1},
                {"attack": "defocus_blur", "severity": 2},
                {"attack": "pixelate", "severity": 3},
            ],
        },
        {
            "preset_id": "adversarial_fgsm",
            "name": "Adversarial Perturbation (FGSM)",
            "task": "detection2d",
            "steps": [{"attack": "fgsm", "severity": 1}],
        },
    ]


@router.post("/attack-recipes/randomize", status_code=201)
async def randomize_recipe(body: RecipeRandomizeIn) -> dict[str, Any]:
    """Generate a randomized attack recipe with N steps or filtered by group."""
    import random
    attacks = load_attacks()
    candidates = [
        item for item in attacks.values()
        if body.group is None or item.group == body.group
    ]
    if not candidates:
        raise HTTPException(status_code=422, detail={"code": "TASK_INCOMPATIBLE", "message": f"no attacks matching group {body.group!r}"})
    selected = random.sample(candidates, k=min(body.n_steps, len(candidates)))
    steps = [
        {"attack_id": attack.name, "severity": random.randint(1, 3)}
        for attack in selected
    ]
    recipe_id = f"recipe-random-{uuid.uuid4().hex[:8]}"
    payload = {
        "id": recipe_id,
        "name": f"Randomized Recipe ({len(steps)} steps)",
        "steps": steps,
    }
    return _store.put_record("attack_recipe", recipe_id, payload)


@router.post("/attack-recipes/sweep", status_code=201)
async def sweep_recipe(body: RecipeSweepIn) -> dict[str, Any]:
    """Generate a parameter sweep recipe across a range of severities."""
    attacks = load_attacks()
    if body.attack_id not in attacks:
        raise HTTPException(status_code=404, detail={"code": "ATTACK_UNKNOWN", "attack_id": body.attack_id})
    steps = [{"attack_id": body.attack_id, "severity": s} for s in body.severity_range]
    recipe_id = f"recipe-sweep-{body.attack_id}-{uuid.uuid4().hex[:8]}"
    payload = {
        "id": recipe_id,
        "name": f"Severity Sweep: {body.attack_id}",
        "steps": steps,
    }
    return _store.put_record("attack_recipe", recipe_id, payload)


@router.post("/attack-recipes/preview")
async def preview_recipe(body: RecipePreviewIn) -> dict[str, Any]:
    """Preview recipe step details and estimated resource cost."""
    recipe_data = None
    if body.recipe_id:
        recipe_data = _store.get_record("attack_recipe", body.recipe_id)
        if recipe_data is None:
            raise HTTPException(status_code=404, detail=f"unknown recipe {body.recipe_id!r}")
    elif body.recipe:
        recipe_data = body.recipe
    else:
        raise HTTPException(status_code=422, detail={"code": "MISSING_RECIPE", "message": "either recipe_id or recipe payload is required"})

    steps = recipe_data.get("steps", [])
    return {
        "preview_id": f"preview-{uuid.uuid4().hex[:8]}",
        "recipe_name": recipe_data.get("name", "Custom Recipe"),
        "step_count": len(steps),
        "steps": steps,
        "estimated_duration_sec_per_sample": round(len(steps) * 0.12, 2),
        "target_modality": "image",
    }



@router.post("/generated-datasets", status_code=202, response_model=GeneratedDatasetJobOut)
async def create_generated_dataset(body: GeneratedDatasetCreateIn) -> GeneratedDatasetJobOut:
    """Queue dataset generation; no attack computation runs in this request."""
    job_id = _generated_datasets.enqueue(body)
    return _generated_job_out(_generated_datasets.get(job_id))


@router.get("/generated-datasets/{job_id}", response_model=GeneratedDatasetJobOut)
async def get_generated_dataset(job_id: str) -> GeneratedDatasetJobOut:
    return _generated_job_out(_generated_datasets.get(job_id))


@router.post("/generated-datasets/{job_id}/cancel", response_model=GeneratedDatasetJobOut)
async def cancel_generated_dataset(job_id: str) -> GeneratedDatasetJobOut:
    if not _workflow_store.request_cancel(job_id):
        raise HTTPException(status_code=404, detail=f"unknown generated dataset job {job_id!r}")
    return _generated_job_out(_generated_datasets.get(job_id))


@router.get("/generated-datasets/{job_id}/manifest", response_model=GeneratedDatasetManifestOut)
async def get_generated_dataset_manifest(job_id: str) -> GeneratedDatasetManifestOut:
    payload = _generated_datasets.manifest(job_id)
    if payload is None:
        _require_generated_job(job_id)
        raise HTTPException(status_code=409, detail="manifest is not available until generation completes")
    return GeneratedDatasetManifestOut(**payload)


@router.get("/generated-datasets/{job_id}/variants", response_model=GeneratedDatasetVariantsOut)
async def get_generated_dataset_variants(job_id: str) -> GeneratedDatasetVariantsOut:
    payload = _generated_datasets.variants(job_id)
    if payload is None:
        _require_generated_job(job_id)
        raise HTTPException(status_code=409, detail="variants are not available until generation completes")
    return GeneratedDatasetVariantsOut(**payload)


@router.get("/generated-datasets/{job_id}/events", response_model=GeneratedDatasetEventsOut)
async def get_generated_dataset_events(job_id: str) -> GeneratedDatasetEventsOut:
    _require_generated_job(job_id)
    return GeneratedDatasetEventsOut(id=job_id, events=_workflow_store.events(job_id))


@router.post("/generated-datasets/{job_id}/validate", response_model=GeneratedDatasetValidationOut)
async def get_generated_dataset_validation(job_id: str) -> GeneratedDatasetValidationOut:
    """Return the worker's persisted validation; routes never recompute artifacts."""
    payload = _generated_datasets.validation(job_id)
    if payload is None:
        _require_generated_job(job_id)
        raise HTTPException(status_code=409, detail="validation is not available until generation completes")
    return GeneratedDatasetValidationOut(**payload)


@router.post("/benchmark/protocols", status_code=201)
async def create_benchmark_protocol(body: dict[str, Any]) -> dict[str, Any]:
    """Create a typed benchmark protocol from request body.

    Generates a content-addressable protocol ID from metric-affecting inputs.
    Protocol starts in DRAFT state; call /validate then /lock to advance.
    """
    from src.evaluation.benchmark_protocol import BenchmarkProtocol

    required = {"dataset_version_id", "model_version_id", "recipe", "task"}
    missing = sorted(required - set(body))
    if missing:
        raise HTTPException(status_code=422, detail={"code": "MISSING_FIELDS", "missing": missing})
    recipe_hash = stable_digest(body.get("recipe", {}), length=40)
    protocol_id = body.get("protocol_id") or f"protocol-{stable_digest(body, length=20)}"
    try:
        protocol = BenchmarkProtocol(
            protocol_id=protocol_id,
            state=body.get("state", "DRAFT"),
            dataset_version_id=body["dataset_version_id"],
            model_version_id=body["model_version_id"],
            task=body.get("task", "detection2d"),
            recipe=body.get("recipe", {}),
            recipe_hash=recipe_hash,
            sample_ids=tuple(body.get("sample_ids", ())),
            sample_hashes=tuple(body.get("sample_hashes", ())),
            dataset_hash=body.get("dataset_hash", ""),
            preprocessing_version=body.get("preprocessing_version", "default"),
            thresholds=body.get("thresholds", {}),
            metric_versions=body.get("metric_versions", {}),
            seed=body.get("seed", 20260730),
            prompt_protocol=body.get("prompt_protocol"),
            class_mapping_version=body.get("class_mapping_version", "1.0.0"),
            bootstrap_config=body.get("bootstrap_config", {}),
            metadata=body.get("metadata", {}),
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_PROTOCOL", "message": str(exc)}) from exc
    payload = protocol.model_dump(mode="json")
    payload["identity_hash"] = protocol.identity_hash()
    return _store.put_record("benchmark_protocol", protocol_id, payload)


@router.post("/benchmark/protocols/{protocol_id}/lock")
async def lock_benchmark_protocol(protocol_id: str) -> dict[str, Any]:
    """Advance a VALIDATED protocol to LOCKED state."""
    from src.evaluation.benchmark_protocol import validate_protocol_transition

    record = _store.get_record("benchmark_protocol", protocol_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown protocol {protocol_id!r}")
    current_state = record.get("state", "DRAFT")
    if not validate_protocol_transition(current_state, "LOCKED"):
        raise HTTPException(
            status_code=409,
            detail={"code": "INVALID_TRANSITION", "current": current_state, "target": "LOCKED"},
        )
    record["state"] = "LOCKED"
    record["identity_hash"] = stable_digest(
        {k: v for k, v in record.items() if k not in {"protocol_id", "state", "identity_hash", "id", "record_type", "created_at", "updated_at", "metadata"}},
        length=40,
    )
    return _store.put_record("benchmark_protocol", protocol_id, record)


@router.get("/benchmark/protocols/{protocol_id}")
async def get_benchmark_protocol(protocol_id: str) -> dict[str, Any]:
    protocol = _store.get_record("benchmark_protocol", protocol_id)
    if protocol is None:
        raise HTTPException(status_code=404, detail=f"unknown protocol {protocol_id!r}")
    return protocol


@router.post("/benchmark/runs", status_code=202, response_model=RunJobOut)
async def create_benchmark_run(config: RunConfig, protocol_id: str = Query(...)) -> RunJobOut:
    """Execute a locked protocol through the same durable async run worker."""
    if _store.get_record("benchmark_protocol", protocol_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown protocol {protocol_id!r}")
    preflight = _runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    run_id = _store.create(config)
    _worker.enqueue(run_id, config)
    return _job_out(_store.get(run_id))


@router.get("/benchmark-runs/{run_id}", response_model=RunJobOut)
async def get_benchmark_run(run_id: str) -> RunJobOut:
    return _job_out(_require_run(run_id))


@router.get("/benchmark-runs/{run_id}/metrics")
async def get_benchmark_metrics(run_id: str) -> dict[str, Any]:
    report = _require_completed_report(run_id)
    return {"run_id": run_id, "metrics": report.get("metrics", {}), "cells": report.get("cells", [])}


@router.get("/benchmark-runs/{run_id}/failures")
async def get_benchmark_failures(run_id: str) -> dict[str, Any]:
    report = _require_completed_report(run_id)
    return {"run_id": run_id, "failures": report.get("worst_cases", []), "skipped": report.get("skipped", [])}


@router.post("/benchmark-runs/{run_id}/cancel", response_model=RunJobOut)
async def cancel_benchmark_run(run_id: str) -> RunJobOut:
    """Cancel a running benchmark job."""
    return await cancel_run(run_id)


@router.websocket("/benchmark-runs/{run_id}/events/ws")
async def benchmark_run_events(run_id: str, websocket: WebSocket) -> None:
    """WebSocket stream for benchmark run events."""
    await run_events(run_id, websocket)



@router.post("/comparisons")
async def compare_runs(baseline_run_id: str = Query(...), candidate_run_id: str = Query(...)) -> dict[str, Any]:
    """Compare completed reports without inventing paired scientific evidence."""
    baseline, candidate = _require_run(baseline_run_id), _require_run(candidate_run_id)
    if baseline["report"] is None or candidate["report"] is None:
        raise HTTPException(status_code=409, detail="both runs must complete before comparison")
    left, right = baseline["report"], candidate["report"]
    paired = left["dataset"] == right["dataset"] and left["n_samples"] == right["n_samples"]
    return {"baseline_run_id": baseline_run_id, "candidate_run_id": candidate_run_id,
            "paired": paired, "clean_score_delta": right["ap_clean"] - left["ap_clean"],
            "cell_count": {"baseline": len(left.get("cells", [])), "candidate": len(right.get("cells", []))}}


@router.get("/catalog/attacks", response_model=list[AttackCatalogItem])
async def list_attacks(
    group: str | None = Query(default=None, min_length=1, max_length=1),
    cost_class: str | None = None,
    modality: str | None = None,
) -> list[AttackCatalogItem]:
    items = [attack.describe() for attack in load_attacks().values()]
    for field, wanted in (("group", group), ("cost_class", cost_class), ("modality", modality)):
        if wanted is not None:
            items = [item for item in items if item[field] == wanted]
    return [AttackCatalogItem(**item) for item in items]


@router.get("/catalog/models", response_model=list[ModelCatalogItem])
async def list_models() -> list[ModelCatalogItem]:
    return [ModelCatalogItem(**adapter.describe()) for adapter in load_adapters().values()]


@router.get("/model-versions", response_model=list[ModelVersionOut])
async def list_model_versions() -> list[ModelVersionOut]:
    """Expose discovered local checkpoints with lineage and safety status."""
    versions = scan_model_artifacts(Path(get_settings().runs_root))
    return [ModelVersionOut.from_domain(version) for version in versions]


@router.get("/model-versions/{version_id}", response_model=ModelVersionOut)
async def get_model_version(version_id: str) -> ModelVersionOut:
    """Get detail for a specific model version."""
    versions = scan_model_artifacts(Path(get_settings().runs_root))
    version = next((v for v in versions if v.id == version_id), None)
    if version is None:
        raise HTTPException(status_code=404, detail=f"unknown model version {version_id!r}")
    return ModelVersionOut.from_domain(version)


@router.get("/model-versions/{version_id}/lineage", response_model=LineageGraphOut)
async def get_model_version_lineage(version_id: str) -> LineageGraphOut:
    """Return parent lineage chain and child model versions."""
    versions = scan_model_artifacts(Path(get_settings().runs_root))
    version = next((v for v in versions if v.id == version_id), None)
    if version is None:
        raise HTTPException(status_code=404, detail=f"unknown model version {version_id!r}")
    children = [v.id for v in versions if v.parent_id == version_id]
    return LineageGraphOut(
        version_id=version.id,
        parent_id=version.parent_id,
        parent_lineage=list(version.parent_lineage),
        children=children,
    )


@router.get("/model-versions/{version_id}/benchmark-history")
async def get_model_version_benchmark_history(version_id: str) -> list[dict[str, Any]]:
    """Return historical benchmark run reports for a model version."""
    history = []
    for run in _store.jobs.values():
        report = run.get("report")
        if report and report.get("model") == version_id:
            history.append({
                "run_id": run["run_id"],
                "created_at": run.get("created_at"),
                "ap_clean": report.get("ap_clean"),
                "n_samples": report.get("n_samples"),
            })
    return history


@router.get("/model-versions/{version_id}/gate-evidence")
async def get_model_version_gate_evidence(version_id: str) -> dict[str, Any]:
    """Return checkpoint gate outcome and evidence records for a model version."""
    gate = _store.get_record("checkpoint_gate", version_id)
    if gate is None:
        # Search by checkpoint hash or model_version_id
        for record in _store.records.values():
            if record.get("record_type") == "checkpoint_gate" and record.get("payload", {}).get("model_version_id") == version_id:
                gate = record.get("payload")
                break
    if gate is None:
        return {"version_id": version_id, "gate_passed": False, "evidence": None, "status": "NO_GATE_EVALUATED"}
    return {"version_id": version_id, "gate_passed": gate.get("passed", False), "evidence": gate}



@router.get("/perception-modes", response_model=list[PerceptionModeOut])
async def list_perception_modes() -> list[PerceptionModeOut]:
    """Product-facing mode availability, including the honest SAM handoff gate."""
    versions = scan_model_artifacts(Path(get_settings().runs_root))
    sam = next((version for version in versions if version.task == "segmentation"), None)
    return [
        PerceptionModeOut(
            id="detection2d",
            title="2D Object Detection, YOLO11",
            metric_labels=(
                "Clean Detection Score",
                "Score After Attack",
                "Performance Lost",
                "Objects Broken by Attack",
                "Overall Robustness",
            ),
            runnable=True,
        ),
        PerceptionModeOut(
            id="segmentation",
            title="Object Segmentation, SAM2",
            metric_labels=(
                "Clean Mask Accuracy",
                "Mask Accuracy After Attack",
                "Mask Performance Lost",
                "Boundary Accuracy",
                "Masks Broken by Attack",
            ),
            runnable=bool(sam and sam.runnable),
            blocked_reason=None if sam and sam.runnable else "WAITING_FOR_ARTIFACTS",
        ),
    ]


@router.post("/attack-recipes/validate", response_model=RecipeValidationOut)
async def validate_recipe(body: RecipeValidationIn) -> RecipeValidationOut:
    """Validate compatibility and resource caps before enqueueing a run."""
    load_attacks()
    result = RecipeBuilder().validate(
        body.recipe,
        ATTACK_CATALOG,
        task=body.task,
        model_capabilities=body.model_capabilities,
        annotation_types=body.annotation_types,
        modality=body.modality,
        online=body.online,
        requested_variants=body.requested_variants,
        bytes_per_variant=body.bytes_per_variant,
    )
    return RecipeValidationOut(
        valid=result.valid,
        errors=result.errors,
        warnings=result.warnings,
        estimate=result.estimate.model_dump(mode="json"),
    )


@router.get("/catalog/datasets", response_model=list[DatasetCatalogItem])
async def list_datasets() -> list[DatasetCatalogItem]:
    return [DatasetCatalogItem(**dataset.describe()) for dataset in load_datasets().values()]


@router.post("/runs/estimate", response_model=CostEstimateOut)
async def estimate_run(config: RunConfig) -> CostEstimateOut:
    return CostEstimateOut(**_runner.estimate(config).as_dict())


@router.post("/runs/preflight", response_model=PreflightOut)
async def preflight_run(config: RunConfig) -> PreflightOut:
    return PreflightOut(**_runner.preflight(config).as_dict())


@router.post("/runs", status_code=202, response_model=RunJobOut)
async def create_run(config: RunConfig) -> RunJobOut:
    """Persist and enqueue a run. Heavy model work never runs in the request."""
    preflight = _runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    run_id = _store.create(config)
    _worker.enqueue(run_id, config)
    return _job_out(_store.get(run_id))


@router.get("/runs", response_model=list[RunJobOut])
async def list_runs() -> list[RunJobOut]:
    return [_job_out(item) for item in _store.list()]


@router.get("/runs/{run_id}", response_model=RunJobOut)
async def get_run(run_id: str) -> RunJobOut:
    return _job_out(_require_run(run_id))


@router.get("/runs/{run_id}/report", response_model=RunReportOut)
async def get_report(run_id: str) -> RunReportOut:
    item = _require_run(run_id)
    if item["report"] is None:
        raise HTTPException(status_code=409, detail=f"run {run_id!r} is {item['status'].lower()}")
    return RunReportOut(**item["report"])


@router.get("/runs/{run_id}/samples")
async def list_samples(
    run_id: str,
    attack: str | None = None,
    severity: int | None = None,
) -> list[dict[str, Any]]:
    item = _require_run(run_id)
    report = item["report"]
    if report is None:
        raise HTTPException(status_code=409, detail="sample evidence is not available until the run completes")
    return [
        sample
        for sample in report.get("sample_results", [])
        if (attack is None or sample["attack"] == attack) and (severity is None or sample["severity"] == severity)
    ]


@router.post("/runs/{run_id}/cancel", response_model=RunJobOut)
async def cancel_run(run_id: str) -> RunJobOut:
    if not _store.request_cancel(run_id):
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return _job_out(_require_run(run_id))


@router.post("/runs/{run_id}/flag-reviews")
async def auto_flag_reviews(run_id: str, threshold: float = Query(default=30.0)) -> dict[str, Any]:
    """Auto-create review items for cells exceeding degradation threshold."""
    _require_run(run_id)
    created = _store.auto_flag_reviews(run_id, threshold=threshold)
    return {"run_id": run_id, "created_reviews": created, "count": len(created)}


@router.websocket("/runs/{run_id}/events/ws")
async def run_events(run_id: str, websocket: WebSocket) -> None:
    await websocket.accept()
    if _store.get(run_id) is None:
        await websocket.send_json({"error": "unknown run"})
        await websocket.close(code=4404)
        return
    cursor = 0
    try:
        while True:
            for event in _store.events(run_id, cursor):
                cursor = event["event_id"]
                await websocket.send_json(event)
            status = _store.get(run_id)
            if status and status["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


# ---- Review endpoints ----

@router.get("/reviews", response_model=list[ReviewOut])
async def list_reviews(status: str | None = Query(default=None)) -> list[ReviewOut]:
    rows = _store.list_reviews(status=status)
    return [ReviewOut(**row) for row in rows]


@router.get("/reviews/{review_id}", response_model=ReviewOut)
async def get_review(review_id: str) -> ReviewOut:
    row = _store.get_review(review_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"unknown review {review_id!r}")
    return ReviewOut(**row)


@router.post("/reviews", status_code=201, response_model=ReviewOut)
async def create_review(body: CreateReviewIn) -> ReviewOut:
    review_id = _store.create_review(
        run_id=body.run_id,
        attack=body.attack,
        severity=body.severity,
        degradation=body.degradation,
        dataset=body.dataset,
        model=body.model,
        flagged_by=body.flagged_by,
        notes=body.notes,
    )
    row = _store.get_review(review_id)
    return ReviewOut(**row)


@router.patch("/reviews/{review_id}", response_model=ReviewOut)
async def resolve_review(review_id: str, body: ResolveReviewIn) -> ReviewOut:
    success = _store.resolve_review(review_id, body.decision, body.decision_note, body.resolved_by)
    if not success:
        raise HTTPException(status_code=404, detail=f"unknown review {review_id!r}")
    row = _store.get_review(review_id)
    return ReviewOut(**row)


@router.get("/failure-cases")
async def list_failure_cases(run_id: str | None = Query(default=None)) -> list[dict[str, Any]]:
    """Aggregate failure cases across completed benchmark runs or query by run_id."""
    failures = []
    runs = [_require_run(run_id)] if run_id else list(_store.jobs.values())
    for run in runs:
        report = run.get("report")
        if report and isinstance(report.get("worst_cases"), list):
            for case in report["worst_cases"]:
                if isinstance(case, dict):
                    failures.append({"run_id": run.get("run_id"), **case})
    return failures


@router.get("/failure-clusters")
async def list_failure_clusters() -> list[dict[str, Any]]:
    """List all persisted failure clusters."""
    records = _store.list_records("failure_cluster")
    return records


@router.post("/failure-clusters", status_code=201)
async def create_failure_cluster(body: FailureClusterCreateIn) -> dict[str, Any]:
    """Persist a failure cluster grouping related failure cases."""
    cluster_id = body.cluster_id or f"cluster-{uuid.uuid4().hex[:8]}"
    payload = {
        "cluster_id": cluster_id,
        "name": body.name,
        "member_ids": body.member_ids,
        "defense_profile_id": body.defense_profile_id,
        "selection_allowed": True,
    }
    return _store.put_record("failure_cluster", cluster_id, payload)


@router.get("/failure-clusters/{cluster_id}")
async def get_failure_cluster(cluster_id: str) -> dict[str, Any]:
    """Get detail for a specific failure cluster."""
    cluster = _store.get_record("failure_cluster", cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"unknown failure cluster {cluster_id!r}")
    return cluster



# ---- Closed-Loop Training ----

@router.post("/closed-loop/start", status_code=201, response_model=ClosedLoopSnapshotOut)
async def start_closed_loop(body: ClosedLoopStartIn) -> dict[str, Any]:
    """Start a closed-loop retraining pipeline from a completed benchmark run."""
    from src.training.closed_loop import ClosedLoopTracker

    item = _store.get(body.run_id)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "RUN_UNKNOWN", "run_id": body.run_id},
        )
    if item["report"] is None:
        raise HTTPException(
            status_code=409,
            detail={"code": "RUN_NOT_COMPLETED", "run_id": body.run_id},
        )

    reported_cases = item["report"].get("worst_cases", [])
    failures = [
        failure
        for failure in reported_cases
        if isinstance(failure, dict) and _is_failure_case(failure)
    ]
    if not failures:
        raise HTTPException(
            status_code=409,
            detail={"code": "NO_FAILURE_CASES", "run_id": body.run_id},
        )

    audit: list[dict[str, Any]] = []
    failure_ids: list[str] = []
    for index, failure in enumerate(failures):
        failure_payload = failure if isinstance(failure, dict) else {"value": failure}
        failure_id = str(
            failure_payload.get("case_id")
            or failure_payload.get("failure_id")
            or f"failure-{stable_digest(failure_payload, length=20)}"
        )
        failure_ids.append(failure_id)
        audit.append(
            {
                "step": index,
                "state": "FAILURE_IDENTIFIED",
                "artifact_id": failure_id,
                "artifact_type": "failure_case",
                "artifact_hash": stable_digest(failure_payload, length=64),
                "parent_step": None,
                "metadata": {"source_run_id": body.run_id},
            }
        )

    loop_id = _workflow_store.create_job(
        "closed_loop", {"source_run_id": body.run_id}
    )
    tracker = ClosedLoopTracker(loop_id=loop_id)
    _workflow_store.append_event(
        loop_id,
        tracker.state,
        {"source_run_id": body.run_id, "progress_ratio": 0.0},
    )
    job = _workflow_store.get_job(loop_id)
    payload = {
        "loop_id": loop_id,
        "source_run_id": body.run_id,
        "state": tracker.state,
        "audit": audit,
        "artifacts": {
            "source_benchmark_run": body.run_id,
            "failure_cases": failure_ids,
        },
        "events": _workflow_store.events(loop_id),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    _workflow_store.checkpoint(loop_id, payload)
    return payload


@router.get("/closed-loop/{loop_id}", response_model=ClosedLoopSnapshotOut)
async def get_closed_loop(loop_id: str) -> dict[str, Any]:
    """Get current state of a closed-loop pipeline."""
    job = _workflow_store.get_job(loop_id)
    if job is None or job["job_type"] != "closed_loop":
        raise HTTPException(
            status_code=404,
            detail={"code": "CLOSED_LOOP_UNKNOWN", "loop_id": loop_id},
        )
    checkpoints = _workflow_store.checkpoints(loop_id)
    if not checkpoints:
        raise HTTPException(
            status_code=409,
            detail={"code": "CLOSED_LOOP_SNAPSHOT_MISSING", "loop_id": loop_id},
        )
    return checkpoints[-1]["payload"]


@router.post("/closed-loop/{loop_id}/advance", response_model=ClosedLoopSnapshotOut)
async def advance_closed_loop(loop_id: str, body: ClosedLoopAdvanceIn) -> dict[str, Any]:
    """Advance a recovery loop to a target state if supported by valid persisted evidence."""
    from src.training.closed_loop import ClosedLoopAuditEntry, ClosedLoopTracker, validate_transition

    job = _workflow_store.get_job(loop_id)
    if job is None or job["job_type"] != "closed_loop":
        raise HTTPException(
            status_code=404,
            detail={"code": "CLOSED_LOOP_UNKNOWN", "loop_id": loop_id},
        )
    checkpoints = _workflow_store.checkpoints(loop_id)
    if not checkpoints:
        raise HTTPException(
            status_code=409,
            detail={"code": "CLOSED_LOOP_SNAPSHOT_MISSING", "loop_id": loop_id},
        )
    latest = checkpoints[-1]["payload"]
    current_state = latest["state"]

    if not validate_transition(current_state, body.target):
        raise HTTPException(
            status_code=409,
            detail={"code": "INVALID_CLOSED_LOOP_TRANSITION", "current": current_state, "target": body.target},
        )

    # Validate evidence exists for the target state
    artifact_type_map = {
        "CLUSTER_FORMED": "failure_cluster",
        "BACKLOG_CREATED": "retraining_backlog",
        "BACKLOG_APPROVED": "retraining_backlog",
        "DEFENSE_PROFILED": "defense_profile",
        "DATASET_MANIFEST_CREATED": "training_dataset_manifest",
        "TRAINING_STARTED": "training_job",
        "TRAINING_COMPLETED": "training_job",
        "CHECKPOINT_VALIDATED": "validated_checkpoint",
        "GATE_EVALUATED": "checkpoint_gate",
        "MODEL_REGISTERED": "registered_model",
        "RE_BENCHMARK_STARTED": "re_benchmark_run",
        "RE_BENCHMARK_COMPLETED": "re_benchmark_run",
        "RECOVERY_REPORTED": "recovery_report",
    }
    art_type = artifact_type_map.get(body.target, "artifact")

    # Perform verification
    valid_evidence = True
    if body.target == "CLUSTER_FORMED":
        cluster = _store.get_record("failure_cluster", body.artifact_id)
        if not cluster:
            valid_evidence = False
    elif body.target in {"BACKLOG_CREATED", "BACKLOG_APPROVED"}:
        backlog = _workflow_store.get_backlog(body.artifact_id)
        if not backlog:
            valid_evidence = False
        elif body.target == "BACKLOG_APPROVED" and backlog.get("status") != "APPROVED":
            valid_evidence = False
    elif body.target == "DEFENSE_PROFILED":
        profile = _store.get_record("defense_profile", body.artifact_id)
        if not profile:
            valid_evidence = False
    elif body.target == "DATASET_MANIFEST_CREATED":
        manifest = _store.get_record("training_dataset_manifest", body.artifact_id)
        if not manifest:
            valid_evidence = False
    elif body.target in {"TRAINING_STARTED", "TRAINING_COMPLETED"}:
        tr_job = _workflow_store.get_job(body.artifact_id)
        if not tr_job:
            valid_evidence = False
        elif body.target == "TRAINING_COMPLETED" and tr_job.get("status") != "COMPLETED":
            valid_evidence = False
    elif body.target == "CHECKPOINT_VALIDATED":
        if not body.artifact_id:
            valid_evidence = False
    elif body.target == "GATE_EVALUATED":
        gate = _store.get_record("checkpoint_gate", body.artifact_id)
        if not gate:
            valid_evidence = False
    elif body.target == "MODEL_REGISTERED":
        if not body.artifact_id:
            valid_evidence = False
    elif body.target in {"RE_BENCHMARK_STARTED", "RE_BENCHMARK_COMPLETED"}:
        bm_run = _store.get(body.artifact_id)
        if not bm_run:
            valid_evidence = False
        elif body.target == "RE_BENCHMARK_COMPLETED" and (not bm_run.get("report")):
            valid_evidence = False
    elif body.target == "RECOVERY_REPORTED":
        comparison = _store.get_record("model_comparison", body.artifact_id)
        if not comparison:
            valid_evidence = False

    if not valid_evidence:
        raise HTTPException(
            status_code=409,
            detail={"code": "INVALID_CLOSED_LOOP_TRANSITION", "current": current_state, "target": body.target, "artifact_id": body.artifact_id},
        )

    # Reconstruct audit list
    audit_entries = []
    for step_idx, entry_dict in enumerate(latest.get("audit", [])):
        audit_entries.append(
            ClosedLoopAuditEntry(
                step=entry_dict.get("step", step_idx),
                state=entry_dict["state"],
                artifact_id=entry_dict["artifact_id"],
                artifact_type=entry_dict.get("artifact_type", "artifact"),
                artifact_hash=entry_dict.get("artifact_hash"),
                parent_step=entry_dict.get("parent_step"),
                metadata=entry_dict.get("metadata", {}),
            )
        )

    tracker = ClosedLoopTracker(
        loop_id=loop_id,
        state=current_state,
        audit=audit_entries,
        artifacts=dict(latest.get("artifacts", {})),
    )

    art_hash = stable_digest({"id": body.artifact_id, "target": body.target}, length=64)
    tracker.advance(
        body.target,
        artifact_id=body.artifact_id,
        artifact_type=art_type,
        artifact_hash=art_hash,
    )

    _workflow_store.append_event(
        loop_id,
        tracker.state,
        {"artifact_id": body.artifact_id, "artifact_type": art_type},
    )

    audit_payloads = [entry.model_dump(mode="json") for entry in tracker.audit]
    updated_payload = {
        "loop_id": loop_id,
        "source_run_id": latest["source_run_id"],
        "state": tracker.state,
        "audit": audit_payloads,
        "artifacts": dict(tracker.artifacts),
        "events": _workflow_store.events(loop_id),
        "created_at": latest["created_at"],
        "updated_at": job["updated_at"],
    }
    _workflow_store.checkpoint(loop_id, updated_payload)
    return updated_payload



# ---- Evidence Status ----

@router.get("/status/evidence", response_model=EvidenceStatusOut)
async def get_evidence_status() -> dict[str, Any]:
    """Aggregate evidence status across all components.

    Single source of truth for which components have passed which evidence tiers.
    """
    placeholders = list_known_versions()
    discovered = await asyncio.to_thread(
        scan_model_artifacts, Path(get_settings().runs_root)
    )
    model_versions_by_id = {version.id: version for version in placeholders}
    placeholders_by_role = {
        str(version.training_metadata.get("role")): version
        for version in placeholders
        if version.training_metadata.get("role")
    }
    for version in discovered:
        role = str(version.training_metadata.get("role", ""))
        placeholder = placeholders_by_role.get(role)
        if placeholder:
            model_versions_by_id.pop(version.id, None)
            version = replace(
                version,
                id=placeholder.id,
                parent_id=placeholder.parent_id,
                parent_lineage=placeholder.parent_lineage,
            )
        model_versions_by_id[version.id] = version

    model_versions = sorted(model_versions_by_id.values(), key=lambda item: item.id)
    components: dict[str, dict[str, Any]] = {}

    for v in model_versions:
        placeholder = placeholders_by_role.get(str(v.training_metadata.get("role", "")))
        components[v.id] = {
            "model_name": v.model_name,
            "task": v.task,
            "runnable": v.runnable,
            "blocked_reason": v.blocked_reason,
            "checkpoint_validated": v.checkpoint_validated,
            "gate_outcome": v.gate_outcome,
            "evidence_tier": v.evidence_tier,
            "parent_id": v.parent_id or (placeholder.parent_id if placeholder else None),
            "parent_lineage": list(
                v.parent_lineage
                or (placeholder.parent_lineage if placeholder else ())
            ),
        }

    # Aggregate summary
    total = len(components)
    verified = sum(1 for c in components.values() if c.get("evidence_tier"))
    waiting = sum(1 for c in components.values() if c.get("blocked_reason") == "WAITING_FOR_ARTIFACTS")
    runnable = sum(1 for c in components.values() if c.get("runnable"))

    return {
        "summary": {
            "total_components": total,
            "verified": verified,
            "waiting_for_artifacts": waiting,
            "runnable": runnable,
            "completion_ratio": round(verified / max(1, total), 4),
        },
        "components": components,
        "evidence_tiers": [
            "CPU_CONTRACT_VERIFIED",
            "REAL_MODEL_VERIFIED",
            "EXTERNAL_VERIFIED",
            "E2E_VERIFIED",
            "SCIENTIFIC_VERIFIED",
            "WAITING_FOR_ARTIFACTS",
        ],
    }


# ---- Helpers ----

def _is_failure_case(payload: dict[str, Any]) -> bool:
    """Accept only benchmark samples carrying an explicit failure signal."""
    degradation = payload.get("degradation_hint")
    has_positive_degradation = (
        isinstance(degradation, (int, float))
        and not isinstance(degradation, bool)
        and degradation > 0.0
    )
    has_reason = any(
        isinstance(payload.get(field), str) and bool(payload[field].strip())
        for field in ("reason", "failure_reason")
    )
    return has_positive_degradation or has_reason or payload.get("failed") is True

def _mean_attack_score(report: dict[str, Any]) -> float:
    cells = report.get("cells", [])
    if not cells:
        return float(report["ap_clean"])
    return sum(float(cell["ap"]) for cell in cells) / len(cells)


def _require_run(run_id: str) -> dict[str, Any]:
    item = _store.get(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
    return item


def _require_generated_job(job_id: str) -> dict[str, Any]:
    item = _generated_datasets.get(job_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"unknown generated dataset job {job_id!r}")
    return item


def _generated_job_out(item: dict[str, Any] | None) -> GeneratedDatasetJobOut:
    if item is None:
        raise HTTPException(status_code=404, detail="unknown generated dataset job")
    result = item["result"] or {}
    return GeneratedDatasetJobOut(
        id=item["id"],
        status=item["status"],
        job_type=item["job_type"],
        error=item["error"],
        cancel_requested=item["cancel_requested"],
        descriptor=result.get("descriptor"),
        artifact_root=result.get("artifact_root"),
    )


def _require_completed_report(run_id: str) -> dict[str, Any]:
    item = _require_run(run_id)
    if item["report"] is None:
        raise HTTPException(status_code=409, detail=f"run {run_id!r} has no completed report")
    return item["report"]


def _job_out(item: dict[str, Any] | None) -> RunJobOut:
    if item is None:
        raise HTTPException(status_code=404, detail="unknown run")
    report = RunReportOut(**item["report"]) if item["report"] else None
    return RunJobOut(
        run_id=item["run_id"],
        status=item["status"],
        progress=item["progress"],
        detail=item["detail"],
        report=report,
        error=item["error"],
    )
