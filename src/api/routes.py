"""AdverTest API: catalog plus durable, asynchronous test-run jobs."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect

from src.adapters import load_adapters
from src.api.generated_dataset_service import GeneratedDatasetService
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.schemas import (
    AttackCatalogItem,
    CostEstimateOut,
    CreateReviewIn,
    DatasetCatalogItem,
    DatasetImportIn,
    GeneratedDatasetCreateIn,
    GeneratedDatasetEventsOut,
    GeneratedDatasetJobOut,
    GeneratedDatasetManifestOut,
    GeneratedDatasetValidationOut,
    GeneratedDatasetVariantsOut,
    ModelCatalogItem,
    ModelComparisonIn,
    ModelVersionOut,
    PerceptionModeOut,
    PreflightOut,
    RecipeRecordIn,
    RecipeValidationIn,
    RecipeValidationOut,
    ResolveReviewIn,
    ReviewOut,
    RunJobOut,
    RunReportOut,
)
from src.api.workflow_store import WorkflowJobStore
from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.recipes import RecipeBuilder
from src.config import get_settings
from src.core.hashing import stable_digest
from src.datasets import load_datasets
from src.datasets.folder import FolderDataset
from src.datasets.versioning import DatasetIngestor, IngestConfig
from src.evaluation.export import export_comparison
from src.models import scan_model_artifacts
from src.pipeline import RunConfig, TestRunner
from src.training.contracts import DefenseProfile

router = APIRouter()
_runner = TestRunner()
_store = SqliteRunStore(get_settings().database_url)
_worker = LocalRunWorker(_store, max_workers=get_settings().worker_max_concurrency)
_workflow_store = WorkflowJobStore(get_settings().database_url)
_generated_datasets = GeneratedDatasetService(
    _workflow_store,
    _store,
    get_settings().artifact_root,
    max_workers=get_settings().worker_max_concurrency,
)
_generated_datasets.recover()
for _run_id, _config in _store.recoverable():
    _worker.enqueue(_run_id, _config)


@router.post("/uploads/images", status_code=201)
async def upload_image(request: Request) -> dict[str, str | int]:
    """Store one user-supplied image; clients may send raw bytes, no multipart needed."""
    filename = request.headers.get("x-filename", "upload.bin")
    filename = Path(filename).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename):
        raise HTTPException(status_code=422, detail="invalid x-filename")
    payload = await request.body()
    if not payload or len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="image payload must be between 1 byte and 25 MiB")
    target_dir = _store.path.parent / "uploads"
    target_dir.mkdir(parents=True, exist_ok=True)
    stored = target_dir / f"{uuid.uuid4().hex}-{filename}"
    stored.write_bytes(payload)
    return {"upload_id": stored.stem, "filename": filename, "path": str(stored), "bytes": len(payload)}


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


@router.post("/attack-recipes", status_code=201)
async def create_recipe(body: RecipeRecordIn) -> dict[str, Any]:
    return _store.put_record("attack_recipe", body.id, body.model_dump(mode="json"))


@router.get("/attack-recipes/{recipe_id}")
async def get_recipe(recipe_id: str) -> dict[str, Any]:
    recipe = _store.get_record("attack_recipe", recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"unknown recipe {recipe_id!r}")
    return recipe


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
    """Lock benchmark inputs before a run, retaining an auditable protocol ID."""
    required = {"dataset_version_id", "model_version_id", "recipe"}
    missing = sorted(required - set(body))
    if missing:
        raise HTTPException(status_code=422, detail={"missing": missing})
    protocol_id = f"protocol-{stable_digest(body, length=20)}"
    protocol = {"protocol_id": protocol_id, "locked": True, **body}
    return _store.put_record("benchmark_protocol", protocol_id, protocol)


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


# ---- Helpers ----

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
