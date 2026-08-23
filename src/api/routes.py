"""AdverTest API: catalog plus durable, asynchronous test-run jobs."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError

from src.adapters import get_adapter
from src.api.dependencies import (
    get_checkpoint_validations,
    get_dataset_import_workers,
    get_generated_datasets,
    get_runner,
    get_store,
    get_training_jobs,
    get_worker,
    get_workflow_store,
)
from src.api.schemas import (
    ClosedLoopAdvanceIn,
    ClosedLoopSnapshotOut,
    ClosedLoopStartIn,
    CreateReviewIn,
    DefenceRunIn,
    EvidenceStatusOut,
    FailureClusterCreateIn,
    GeneratedDatasetJobOut,
    LineageGraphOut,
    ModelComparisonIn,
    ModelFamilyOut,
    ModelVersionOut,
    PerceptionModeOut,
    QuickInferenceIn,
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
from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.recipes import RecipeBuilder
from src.config import get_settings
from src.core.hashing import stable_digest
from src.datasets import load_datasets
from src.evaluation.export import export_comparison
from src.models import list_known_versions, scan_base_checkpoints, scan_model_artifacts
from src.models.families import FAMILIES, adapter_request, approved_model_config
from src.models.versions import ModelVersion
from src.pipeline import RunConfig
from src.training.contracts import DefenseProfile, TrainingRunConfig

router = APIRouter()


class _DynamicDependencyProxy:
    def __init__(self, getter):
        self._getter = getter

    def __getattr__(self, name: str):
        return getattr(self._getter(), name)


_runner = _DynamicDependencyProxy(get_runner)
_store = _DynamicDependencyProxy(get_store)
_worker = _DynamicDependencyProxy(get_worker)
_workflow_store = _DynamicDependencyProxy(get_workflow_store)
_training_jobs = _DynamicDependencyProxy(get_training_jobs)
_generated_datasets = _DynamicDependencyProxy(get_generated_datasets)
_checkpoint_validations = _DynamicDependencyProxy(get_checkpoint_validations)
_dataset_import_workers = _DynamicDependencyProxy(get_dataset_import_workers)


def _validate_uploaded_image(payload: bytes) -> tuple[str, tuple[int, int]]:
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


@router.post("/checkpoints/uploads", status_code=201)
async def upload_checkpoint(request: Request) -> dict[str, Any]:
    """Persist an untrusted checkpoint for a separate validation worker.

    No model deserialization happens in the request process; a checkpoint is
    deliberately non-runnable until its validator records a READY result.
    """
    filename = Path(request.headers.get("x-filename", "checkpoint.pt")).name
    task_id = request.headers.get("x-task-id", "")
    family_id = request.headers.get("x-model-family-id", "")
    model_config_id = request.headers.get("x-model-config-id", "")
    display_name = request.headers.get("x-display-name", filename)
    checkpoint_role = request.headers.get("x-checkpoint-role", "base")
    parent_checkpoint_id = request.headers.get("x-parent-checkpoint-id")
    if checkpoint_role not in {"base", "fine_tuned", "repaired"}:
        raise HTTPException(status_code=422, detail={"code": "CHECKPOINT_ROLE_INVALID"})
    if checkpoint_role != "base" and not parent_checkpoint_id:
        raise HTTPException(status_code=422, detail={"code": "CHECKPOINT_PARENT_REQUIRED"})
    if checkpoint_role == "base" and parent_checkpoint_id:
        raise HTTPException(status_code=422, detail={"code": "BASE_CHECKPOINT_PARENT_FORBIDDEN"})
    family = FAMILIES.get(family_id)
    if family is None or task_id not in family.supported_tasks:
        raise HTTPException(status_code=422, detail={"code": "MODEL_FAMILY_TASK_MISMATCH"})
    if Path(filename).suffix.lower() not in family.checkpoint_extensions:
        raise HTTPException(status_code=422, detail={"code": "CHECKPOINT_EXTENSION_INVALID"})
    model_config = approved_model_config(family_id, model_config_id)
    if family_id == "pointpillars3d" and model_config is None:
        raise HTTPException(status_code=422, detail={"code": "MODEL_FAMILY_CONFIG_INVALID"})
    checkpoint_id = f"checkpoint-{uuid.uuid4().hex[:16]}"
    target_root = Path(get_settings().checkpoint_root).expanduser().resolve() / "uploaded"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"{checkpoint_id}-{filename}"
    digest = hashlib.sha256()
    total = 0
    with target.open("xb") as stream:
        async for chunk in request.stream():
            total += len(chunk)
            if total > 4 * 1024 * 1024 * 1024:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=422, detail={"code": "CHECKPOINT_TOO_LARGE"})
            digest.update(chunk)
            stream.write(chunk)
    if total == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail={"code": "CHECKPOINT_EMPTY"})
    duplicate = next(
        (item for item in _store.list_records("checkpoint") if item.get("sha256") == digest.hexdigest()),
        None,
    )
    if duplicate is not None:
        target.unlink(missing_ok=True)
        return duplicate
    payload = {
        "checkpoint_id": checkpoint_id,
        "display_name": display_name[:128],
        "task_id": task_id,
        "family_id": family_id,
        "model_config": model_config,
        "storage_path": str(target),
        "sha256": digest.hexdigest(),
        "bytes": total,
        "status": "PENDING_VALIDATION",
        "checkpoint_role": checkpoint_role,
        "parent_checkpoint_id": parent_checkpoint_id,
        "training_dataset_version_id": request.headers.get("x-training-dataset-version-id"),
        "validation": {"state": "PENDING", "reason": "Validation runs outside the API request process."},
    }
    validation_job_id = _workflow_store.create_job("checkpoint_validation", {"checkpoint_id": checkpoint_id})
    payload["validation_job_id"] = validation_job_id
    stored = _store.put_record("checkpoint", checkpoint_id, payload)
    _checkpoint_validations.start(checkpoint_id, validation_job_id)
    return stored


@router.get("/checkpoints")
async def list_checkpoints() -> list[dict[str, Any]]:
    return _store.list_records("checkpoint")


@router.get("/checkpoints/{checkpoint_id}")
async def get_checkpoint(checkpoint_id: str) -> dict[str, Any]:
    checkpoint = _store.get_record("checkpoint", checkpoint_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="CHECKPOINT_UNKNOWN")
    return checkpoint


@router.get("/checkpoints/{checkpoint_id}/validation-events")
async def checkpoint_validation_events(checkpoint_id: str) -> list[dict[str, Any]]:
    checkpoint = _store.get_record("checkpoint", checkpoint_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="CHECKPOINT_UNKNOWN")
    job_id = checkpoint.get("validation_job_id")
    return _workflow_store.events(str(job_id)) if job_id else []


@router.post("/checkpoints/{checkpoint_id}/validation/cancel")
async def cancel_checkpoint_validation(checkpoint_id: str) -> dict[str, Any]:
    checkpoint = _store.get_record("checkpoint", checkpoint_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="CHECKPOINT_UNKNOWN")
    job_id = checkpoint.get("validation_job_id")
    if not job_id or not _workflow_store.request_cancel(str(job_id)):
        raise HTTPException(status_code=409, detail="CHECKPOINT_VALIDATION_NOT_CANCELLABLE")
    return {"checkpoint_id": checkpoint_id, "job_id": job_id, "state": "CANCEL_REQUESTED"}


@router.websocket("/jobs/{job_id}/events/ws")
async def workflow_job_events(job_id: str, websocket: WebSocket) -> None:
    """Shared WebSocket feed for import/validation/generation/training jobs."""
    await websocket.accept()
    if _workflow_store.get_job(job_id) is None:
        await websocket.send_json({"error": "WORKFLOW_JOB_UNKNOWN"})
        await websocket.close(code=4404)
        return
    cursor = 0
    try:
        while True:
            events = _workflow_store.events(job_id)
            for event in events[cursor:]:
                cursor += 1
                await websocket.send_json(event)
            job = _workflow_store.get_job(job_id)
            if job and job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


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
    baseline_signature = _comparison_signature(left)
    candidate_signature = _comparison_signature(right)
    paired = baseline_signature == candidate_signature
    comparison_id = f"comparison-{stable_digest(body.model_dump(mode='json'), length=20)}"
    metric_deltas = _comparison_metric_deltas(left, right) if paired else {}
    baseline_attack_score = _mean_attack_score(left)
    candidate_attack_score = _mean_attack_score(right)
    lost_score = left["ap_clean"] - baseline_attack_score
    recovered_score = candidate_attack_score - baseline_attack_score
    recovery_ratio = None if not paired or lost_score <= 0 else recovered_score / lost_score
    payload = {
        "comparison_id": comparison_id,
        **body.model_dump(mode="json"),
        "paired": paired,
        "baseline_signature": baseline_signature,
        "candidate_signature": candidate_signature,
        "incompatibilities": []
        if paired
        else [key for key in baseline_signature if baseline_signature.get(key) != candidate_signature.get(key)],
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


def _comparison_metric_deltas(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, float | str]]:
    """Keep base/candidate metric families explicit rather than one blended score."""
    output: dict[str, dict[str, float | str]] = {
        # Compatibility alias retained for one API cycle.  New consumers use
        # clean_ap50 / clean_ap75 / clean_map50_95 below.
        "clean_detection_score": {
            "value": float(right.get("ap_clean", 0.0)) - float(left.get("ap_clean", 0.0)),
            "unit": "ratio",
        },
    }
    for input_name, payload in (("clean", left), ("attacked", right)):
        # ``attacked`` here is intentionally filled below from each run's final
        # attack cell; no clean/attacked result is silently combined.
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
        headers={
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "X-Content-SHA256": artifact.sha256,
        },
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
    candidates = [item for item in attacks.values() if body.group is None or item.group == body.group]
    if not candidates:
        raise HTTPException(
            status_code=422,
            detail={"code": "TASK_INCOMPATIBLE", "message": f"no attacks matching group {body.group!r}"},
        )
    rng = random.Random(body.seed)
    selected = rng.sample(candidates, k=min(body.n_steps, len(candidates)))
    steps = [{"attack_id": attack.name, "severity": rng.randint(1, 3)} for attack in selected]
    recipe_id = f"recipe-random-{uuid.uuid4().hex[:8]}"
    payload = {
        "id": recipe_id,
        "name": f"Randomized Recipe ({len(steps)} steps)",
        "seed": body.seed,
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
        raise HTTPException(
            status_code=422,
            detail={"code": "MISSING_RECIPE", "message": "either recipe_id or recipe payload is required"},
        )

    steps = recipe_data.get("steps", [])
    return {
        "preview_id": f"preview-{uuid.uuid4().hex[:8]}",
        "recipe_name": recipe_data.get("name", "Custom Recipe"),
        "step_count": len(steps),
        "steps": steps,
        "estimated_duration_sec_per_sample": round(len(steps) * 0.12, 2),
        "target_modality": "image",
    }


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
        {
            k: v
            for k, v in record.items()
            if k
            not in {
                "protocol_id",
                "state",
                "identity_hash",
                "id",
                "record_type",
                "created_at",
                "updated_at",
                "metadata",
            }
        },
        length=40,
    )
    return _store.update_record("benchmark_protocol", protocol_id, record)


@router.get("/benchmark/protocols/{protocol_id}")
async def get_benchmark_protocol(protocol_id: str) -> dict[str, Any]:
    protocol = _store.get_record("benchmark_protocol", protocol_id)
    if protocol is None:
        raise HTTPException(status_code=404, detail=f"unknown protocol {protocol_id!r}")
    return protocol


@router.post("/benchmark/runs", status_code=202, response_model=RunJobOut)
async def create_benchmark_run(config: RunConfig, protocol_id: str = Query(...)) -> RunJobOut:
    """Execute a locked protocol through the same durable async run worker."""
    protocol = _store.get_record("benchmark_protocol", protocol_id)
    if protocol is None:
        raise HTTPException(status_code=404, detail=f"unknown protocol {protocol_id!r}")
    if protocol.get("state") != "LOCKED":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "PROTOCOL_NOT_LOCKED",
                "state": protocol.get("state"),
            },
        )
    config = _resolve_run_config(config).model_copy(update={"benchmark_protocol_id": protocol_id})
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
    record = _require_run(run_id)
    if record["status"] not in ("COMPLETED", "FAILED", "CANCELLED"):
        _store.request_cancel(run_id)
        if record["status"] == "QUEUED":
            _store.fail(run_id, "Cancelled by user", cancelled=True)
    return _job_out(_store.get(run_id) or record)


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
    return {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "paired": paired,
        "clean_score_delta": right["ap_clean"] - left["ap_clean"],
        "cell_count": {"baseline": len(left.get("cells", [])), "candidate": len(right.get("cells", []))},
    }


@router.get("/model-versions", response_model=list[ModelVersionOut])
async def list_model_versions() -> list[ModelVersionOut]:
    """Expose discovered local checkpoints with lineage and safety status."""
    versions = _registered_model_versions()
    return [ModelVersionOut.from_domain(version) for version in versions]


@router.get("/model-families", response_model=list[ModelFamilyOut])
async def list_model_families(task_id: str = Query(...)) -> list[ModelFamilyOut]:
    """Families are filtered by perception task; weights are selected later."""
    return [
        ModelFamilyOut(
            id=family.id,
            display_name=family.display_name,
            supported_tasks=sorted(family.supported_tasks),
            checkpoint_extensions=sorted(family.checkpoint_extensions),
            runnable=family.runnable,
            blocked_reason=family.blocked_reason,
        )
        for family in FAMILIES.values()
        if task_id in family.supported_tasks
    ]


@router.get("/base-checkpoints", response_model=list[ModelVersionOut])
async def list_base_checkpoints(
    task_id: str = Query(...),
    model_family_id: str = Query(...),
) -> list[ModelVersionOut]:
    return [
        ModelVersionOut.from_domain(version)
        for version in _registered_model_versions()
        if version.task == task_id
        and version.model_family_id == model_family_id
        and (version.runnable or version.checkpoint_role == "base")
    ]


@router.get("/defence-checkpoints", response_model=list[ModelVersionOut])
async def list_defence_checkpoints(task_id: str = Query(...)) -> list[ModelVersionOut]:
    return [
        ModelVersionOut.from_domain(version)
        for version in _registered_model_versions()
        if version.task == task_id and version.checkpoint_role != "base"
    ]


@router.get("/model-versions/{version_id}", response_model=ModelVersionOut)
async def get_model_version(version_id: str) -> ModelVersionOut:
    """Get detail for a specific model version."""
    versions = _registered_model_versions()
    version = next((v for v in versions if v.id == version_id), None)
    if version is None:
        raise HTTPException(status_code=404, detail=f"unknown model version {version_id!r}")
    return ModelVersionOut.from_domain(version)


@router.get("/model-versions/{version_id}/lineage", response_model=LineageGraphOut)
async def get_model_version_lineage(version_id: str) -> LineageGraphOut:
    """Return parent lineage chain and child model versions."""
    versions = _registered_model_versions()
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
    for run in _store.list():
        report = run.get("report")
        if report and (report.get("model") == version_id or report.get("model_version_id") == version_id):
            history.append(
                {
                    "run_id": run["run_id"],
                    "created_at": run.get("created_at"),
                    "ap_clean": report.get("ap_clean"),
                    "n_samples": report.get("n_samples"),
                }
            )
    return history


@router.get("/model-versions/{version_id}/gate-evidence")
async def get_model_version_gate_evidence(version_id: str) -> dict[str, Any]:
    """Return checkpoint gate outcome and evidence records for a model version."""
    gate = _store.get_record("checkpoint_gate", version_id)
    if gate is None:
        # Search by checkpoint hash or model_version_id
        for record in _store.list_records("checkpoint_gate"):
            if record.get("model_version_id") == version_id or record.get("id") == version_id:
                gate = record
                break
    if gate is None:
        return {"version_id": version_id, "gate_passed": False, "evidence": None, "status": "NO_GATE_EVALUATED"}
    return {"version_id": version_id, "gate_passed": gate.get("passed", False), "evidence": gate}


@router.get("/perception-modes", response_model=list[PerceptionModeOut])
async def list_perception_modes() -> list[PerceptionModeOut]:
    """Product-facing mode availability, including the honest SAM handoff gate."""
    versions = _registered_model_versions()
    sam = next((version for version in versions if version.task == "segmentation"), None)
    return [
        PerceptionModeOut(
            id="detection2d",
            title="2D Object Detection",
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
            title="Instance Segmentation",
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
        PerceptionModeOut(
            id="detection3d",
            title="3D Object Detection",
            metric_labels=(),
            runnable=False,
            blocked_reason="COMING_LATER",
            status="coming_later",
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


@router.post("/defence-runs", status_code=202, response_model=RunJobOut)
async def create_defence_run(body: DefenceRunIn) -> RunJobOut:
    """Run a reviewed Defence candidate on the baseline's immutable protocol."""
    baseline = _store.get(body.baseline_run_id)
    if baseline is None:
        raise HTTPException(status_code=404, detail={"code": "BASELINE_RUN_UNKNOWN"})
    if baseline.get("status") != "COMPLETED" or baseline.get("report") is None:
        raise HTTPException(status_code=409, detail={"code": "BASELINE_RUN_NOT_COMPLETED"})
    candidate = next((item for item in _registered_model_versions() if item.id == body.checkpoint_id), None)
    if candidate is None:
        raise HTTPException(status_code=404, detail={"code": "CHECKPOINT_UNKNOWN"})
    if candidate.checkpoint_role == "base":
        raise HTTPException(status_code=422, detail={"code": "DEFENCE_CANDIDATE_ROLE_REQUIRED"})
    baseline_config = RunConfig.model_validate(baseline["config"])
    if candidate.task != baseline_config.task_id or candidate.model_family_id != baseline_config.model_family_id:
        raise HTTPException(status_code=422, detail={"code": "DEFENCE_PROTOCOL_TASK_OR_FAMILY_MISMATCH"})
    config = baseline_config.model_copy(
        update={
            "checkpoint_id": candidate.id,
            "model_version_id": candidate.id,
            "model": candidate.model_name,
            "adapter_params": {},
            "execution_mode": "benchmark",
        }
    )
    config = _resolve_run_config(config, allow_defence=True)
    preflight = _runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    run_id = _store.create(config)
    _store.put_record(
        "defence_evaluation",
        run_id,
        {
            "run_id": run_id,
            "baseline_run_id": body.baseline_run_id,
            "checkpoint_id": candidate.id,
            "protocol_config": baseline_config.model_dump(mode="json"),
        },
    )
    _worker.enqueue(run_id, config)
    return _job_out(_store.get(run_id))


@router.post("/inference-experiments", status_code=202, response_model=RunJobOut)
async def create_inference_experiment(body: QuickInferenceIn) -> RunJobOut:
    """Queue a qualitative raw-upload experiment without manufacturing AP/mAP."""
    batch_root = (_upload_root() / body.upload_batch_id).resolve()
    uploads_root = _upload_root()
    if uploads_root not in batch_root.parents or not (batch_root / "images").is_dir():
        print(
            f"DEBUG_DEBUG: uploads_root={uploads_root}, batch_root={batch_root}, is_dir={(batch_root / 'images').is_dir()}"
        )
        import os

        print(f"DEBUG_DEBUG: {os.listdir(uploads_root) if uploads_root.exists() else 'uploads_root missing'}")
        print(f"DEBUG_DEBUG: {os.listdir(batch_root) if batch_root.exists() else 'batch_root missing'}")
        raise HTTPException(status_code=404, detail="UPLOAD_BATCH_UNKNOWN")
    config = RunConfig(
        model_version_id=body.model_version_id,
        model_family_id=body.model_family_id,
        checkpoint_id=body.checkpoint_id,
        task_id=body.task_id,
        dataset="folder_dataset",
        dataset_params={"root": str(batch_root), "input_format": "advertest"},
        recipe=body.recipe,
        attacks=body.attacks,
        severities=body.severities,
        seed=body.seed,
        limit=body.limit,
        iou_threshold=body.iou_threshold,
        confidence_threshold=body.confidence_threshold,
        execution_mode="quick_inference",
    )
    config = _resolve_run_config(config)
    preflight = _runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})
    run_id = _store.create(config)
    _worker.enqueue(run_id, config)
    return _job_out(_store.get(run_id))


@router.get("/runs/{run_id}/defence-candidates", response_model=list[ModelVersionOut])
async def run_defence_candidates(run_id: str) -> list[ModelVersionOut]:
    """Fine-tuned/repaired weights are visible only after an attack run."""
    run = _require_run(run_id)
    task_id = (run.get("config") or {}).get("task_id", "detection2d")
    return [
        ModelVersionOut.from_domain(version)
        for version in _registered_model_versions()
        if version.task == task_id and version.checkpoint_role != "base"
    ]


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
        _sample_with_artifact_urls(sample)
        for sample in report.get("sample_results", [])
        if (attack is None or sample["attack"] == attack) and (severity is None or sample["severity"] == severity)
    ]


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
    runs = [_require_run(run_id)] if run_id else _store.list()
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
    failures = [failure for failure in reported_cases if isinstance(failure, dict) and _is_failure_case(failure)]
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

    loop_id = _workflow_store.create_job("closed_loop", {"source_run_id": body.run_id})
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
            detail={
                "code": "INVALID_CLOSED_LOOP_TRANSITION",
                "current": current_state,
                "target": body.target,
                "artifact_id": body.artifact_id,
            },
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
    discovered = await asyncio.to_thread(scan_model_artifacts, Path(get_settings().runs_root))
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
            "parent_lineage": list(v.parent_lineage or (placeholder.parent_lineage if placeholder else ())),
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


def _sample_with_artifact_urls(sample: dict[str, Any]) -> dict[str, Any]:
    """Expose only browser-readable artifact URIs; never leak filesystem paths."""
    result = {key: value for key, value in sample.items() if not key.endswith("_path")}
    result["artifacts"] = {
        "clean_input_url": _artifact_uri(sample.get("clean_image_path")),
        "attacked_input_url": _artifact_uri(sample.get("attacked_image_path")),
        "clean_prediction_url": _artifact_uri(sample.get("clean_prediction_path")),
        "attacked_prediction_url": _artifact_uri(sample.get("attacked_prediction_path")),
    }
    return result


def _artifact_uri(path_value: Any) -> str | None:
    if not path_value:
        return None
    candidate = Path(str(path_value)).expanduser().resolve()
    data_root = Path(get_settings().data_root).expanduser().resolve()
    if candidate != data_root and data_root not in candidate.parents:
        return None
    return f"/data/{candidate.relative_to(data_root).as_posix()}"


def _registered_model_versions() -> list[Any]:
    """Keep every planned lineage role visible even when its artifact is absent."""
    placeholders = list_known_versions()
    by_role = {str(item.training_metadata.get("role", "")): item for item in placeholders}
    versions = {item.id: item for item in placeholders}
    for base in scan_base_checkpoints(Path(get_settings().checkpoint_root)):
        versions[base.id] = base
    for discovered in scan_model_artifacts(Path(get_settings().runs_root)):
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
    for record in _store.list_records("checkpoint"):
        if record.get("status") != "READY":
            continue
        task_id = str(record.get("task_id", ""))
        family_id = str(record.get("family_id", ""))
        path = Path(str(record.get("storage_path", ""))).expanduser().resolve()
        family = FAMILIES.get(family_id)
        if family is None or task_id not in family.supported_tasks or not path.is_file():
            continue
        role = str(record.get("checkpoint_role", "base"))
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
            runnable=family.runnable,
            blocked_reason=family.blocked_reason,
            checkpoint_validated=True,
            gate_outcome="PASSED",
            evidence_tier="QUARANTINE_VALIDATED",
            model_family_id=family_id,
            checkpoint_role=role,  # type: ignore[arg-type]
        )
    return sorted(versions.values(), key=lambda item: item.id)


def _attack_catalog_availability(
    *,
    task_id: str | None,
    model_family_id: str | None,
    checkpoint_id: str | None,
    dataset_name: str | None,
) -> tuple[str | None, dict[str, tuple[str, ...]]] | None:
    """Evaluate attack cards against the selected product contracts.

    The catalog remains descriptive until a model family or checkpoint is
    selected.  Once selected, these decisions use the same adapter metadata
    and attack-catalog compatibility contract that governs a real run.
    """
    if model_family_id is None and checkpoint_id is None:
        return None

    version = None
    if checkpoint_id:
        version = next((item for item in _registered_model_versions() if item.id == checkpoint_id), None)
        if version is None:
            return "CHECKPOINT_UNKNOWN", {}
        if model_family_id is not None and version.model_family_id != model_family_id:
            return "CHECKPOINT_FAMILY_MISMATCH", {}
        if task_id is not None and version.task != task_id:
            return "MODEL_FAMILY_TASK_MISMATCH", {}
        if version.checkpoint_role != "base":
            return "DEFENCE_CHECKPOINT_NOT_ALLOWED_IN_ATTACK", {}
        if not version.runnable or not version.checkpoint_path:
            return f"MODEL_NOT_RUNNABLE:{version.blocked_reason or 'CHECKPOINT_MISSING'}", {}
        checkpoint = Path(version.checkpoint_path).resolve()
        if not checkpoint.is_file():
            return "CHECKPOINT_MISSING", {}
        try:
            adapter_name, adapter_params = adapter_request(
                version, checkpoint=str(checkpoint), config=RunConfig(), settings=get_settings()
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


def _upload_root() -> Path:
    root = Path(get_settings().data_root).expanduser().resolve() / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_root() -> Path:
    root = Path(get_settings().data_root).expanduser().resolve() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_run_config(config: RunConfig, *, allow_defence: bool = False) -> RunConfig:
    """Resolve a product ModelVersion server-side; browsers never receive weight paths."""
    checkpoint_id = config.checkpoint_id or config.model_version_id
    if not checkpoint_id:
        return config
    version = next(
        (item for item in _registered_model_versions() if item.id == checkpoint_id),
        None,
    )
    if version is None:
        raise HTTPException(status_code=404, detail={"code": "CHECKPOINT_UNKNOWN", "checkpoint_id": checkpoint_id})
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
    for attack_name in config.attacks:
        attack = load_attacks().get(attack_name)
        allowed = attack.required_tasks or frozenset({"detection2d", "segmentation"})
        if version.task not in allowed:
            raise HTTPException(
                status_code=422,
                detail={"code": "ATTACK_NOT_COMPATIBLE", "attack": attack_name, "task_id": version.task},
            )
    if not version.runnable or not version.checkpoint_path:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MODEL_NOT_RUNNABLE",
                "model_version_id": config.model_version_id,
                "reason": version.blocked_reason,
            },
        )
    checkpoint = Path(version.checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise HTTPException(
            status_code=409, detail={"code": "CHECKPOINT_MISSING", "model_version_id": config.model_version_id}
        )
    settings = get_settings()
    try:
        adapter_name, adapter_params = adapter_request(
            version, checkpoint=str(checkpoint), config=config, settings=settings
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409, detail={"code": str(exc), "model_version_id": config.model_version_id}
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


def _comparison_signature(report: dict[str, Any]) -> dict[str, Any]:
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


def _is_failure_case(payload: dict[str, Any]) -> bool:
    """Accept only benchmark samples carrying an explicit failure signal."""
    degradation = payload.get("degradation_hint")
    has_positive_degradation = (
        isinstance(degradation, (int, float)) and not isinstance(degradation, bool) and degradation > 0.0
    )
    has_reason = any(
        isinstance(payload.get(field), str) and bool(payload[field].strip()) for field in ("reason", "failure_reason")
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
