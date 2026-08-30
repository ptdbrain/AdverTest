"""Defence router: profiles, adversarial training, model comparisons, lineage, and recovery loops."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response, WebSocket, WebSocketDisconnect

from src.api.dependencies import (
    get_runner,
    get_store,
    get_training_jobs,
    get_worker,
    get_workflow_store,
)
from src.api.defense_scope import require_scoped_record, require_scoped_run, require_scoped_workflow_job
from src.api.platform_dependencies import require_project_member
from src.api.helpers import (
    is_failure_case,
    job_out,
    registered_model_versions,
    require_run,
    resolve_run_config,
)
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.schemas import (
    ClosedLoopAdvanceIn,
    ClosedLoopSnapshotOut,
    ClosedLoopStartIn,
    DefenceRunIn,
    FailureClusterCreateIn,
    LineageGraphOut,
    ModelComparisonIn,
    ModelVersionOut,
    RetrainingBacklogIn,
    RetrainingBacklogItemIn,
    RunJobOut,
    TrainingRunIn,
)
from src.api.training_service import TrainingJobService
from src.api.workflow_store import WorkflowJobStore
from src.config import get_settings
from src.core.hashing import stable_digest
from src.evaluation.export import export_comparison
from src.evaluation.defense_report import build_defense_report
from src.models import scan_model_artifacts
from src.pipeline.runner import RunConfig, TestRunner
from src.training.contracts import DefenseProfile, TrainingRunConfig

router = APIRouter(tags=["Defence"])


# ---- Defense Profiles ----


@router.post("/defense-profiles", status_code=201)
async def create_defense_profile(
    body: DefenseProfile,
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Persist a defense profile defining dataset mixing ratios and recipe IDs."""
    return store.put_record("defense_profile", body.profile_id, body.model_dump(mode="json"))


@router.get("/defense-profiles/{profile_id}")
async def get_defense_profile(
    profile_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Retrieve a stored defense profile by ID."""
    profile = store.get_record("defense_profile", profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"unknown defense profile {profile_id!r}")
    return profile


# ---- Training Runs & Manifests ----


@router.post("/training-runs/estimate")
async def estimate_training_run(
    body: TrainingRunIn,
    training_jobs: TrainingJobService = Depends(get_training_jobs),
) -> dict[str, Any]:
    """Estimate a registered trainer without scheduling external training."""
    config = TrainingRunConfig(run_id=f"estimate-{uuid.uuid4().hex}", **body.model_dump(mode="json"))
    try:
        return training_jobs.estimate(config)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"TRAINER_NOT_AVAILABLE: {exc}") from exc


@router.post("/training-runs", status_code=202)
async def start_training_run(
    body: TrainingRunIn,
    training_jobs: TrainingJobService = Depends(get_training_jobs),
) -> dict[str, Any]:
    """Queue training only after the requested parent checkpoint is runnable."""
    version = next(
        (item for item in scan_model_artifacts(Path(get_settings().runs_root)) if item.id == body.model_version),
        None,
    )
    if version is None or not version.runnable:
        raise HTTPException(status_code=409, detail="WAITING_FOR_ARTIFACTS")
    config = TrainingRunConfig(run_id=f"queued-{uuid.uuid4().hex}", **body.model_dump(mode="json"))
    try:
        job_id = training_jobs.enqueue(config)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=f"TRAINER_NOT_AVAILABLE: {exc}") from exc
    return training_jobs.get(job_id) or {"id": job_id, "status": "QUEUED"}


@router.get("/training-runs")
async def list_training_runs(
    training_jobs: TrainingJobService = Depends(get_training_jobs),
) -> list[dict[str, Any]]:
    """List all training run jobs."""
    return training_jobs.list()


@router.get("/training-runs/{job_id}")
async def get_training_run(
    job_id: str,
    training_jobs: TrainingJobService = Depends(get_training_jobs),
) -> dict[str, Any]:
    """Get training job detail and current status."""
    job = training_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="TRAINING_RUN_UNKNOWN")
    return job


@router.post("/training-runs/{job_id}/cancel")
async def cancel_training_run(
    job_id: str,
    training_jobs: TrainingJobService = Depends(get_training_jobs),
) -> dict[str, Any]:
    """Cancel an active or queued training run."""
    if training_jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="TRAINING_RUN_UNKNOWN")
    training_jobs.cancel(job_id)
    return training_jobs.get(job_id) or {"id": job_id, "status": "CANCEL_REQUESTED"}


@router.get("/training-runs/{job_id}/checkpoints")
async def get_training_run_checkpoints(
    job_id: str,
    training_jobs: TrainingJobService = Depends(get_training_jobs),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> list[dict[str, Any]]:
    """Retrieve saved model checkpoints for a training job."""
    if training_jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="TRAINING_RUN_UNKNOWN")
    return workflow_store.checkpoints(job_id)


@router.get("/training-runs/{job_id}/events")
async def get_training_run_events(
    job_id: str,
    cursor: int = Query(default=0, ge=0),
    training_jobs: TrainingJobService = Depends(get_training_jobs),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> list[dict[str, Any]]:
    """Retrieve log events emitted by the training job."""
    if training_jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="TRAINING_RUN_UNKNOWN")
    return workflow_store.events(job_id, cursor=cursor)


@router.get("/training-dataset-manifests/{manifest_id}")
async def get_training_dataset_manifest(
    manifest_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Retrieve an immutable training dataset manifest."""
    manifest = store.get_record("training_dataset_manifest", manifest_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"unknown training dataset manifest {manifest_id!r}")
    return manifest


@router.websocket("/training-runs/{job_id}/events/ws")
async def training_run_events_ws(
    job_id: str,
    websocket: WebSocket,
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> None:
    """Stream training progress events via WebSocket."""
    await websocket.accept()
    if workflow_store.get_job(job_id) is None:
        await websocket.send_json({"error": "unknown training run"})
        await websocket.close(code=4404)
        return
    cursor = 0
    try:
        while True:
            for event in workflow_store.events(job_id, cursor=cursor):
                cursor = event["event_id"]
                await websocket.send_json(event)
            job = workflow_store.get_job(job_id)
            if job and job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return


# ---- Retraining Backlogs ----


@router.post("/retraining-backlogs", status_code=201)
async def create_retraining_backlog(
    body: RetrainingBacklogIn,
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> dict[str, Any]:
    """Create a curated retraining backlog to organize failure fixes."""
    return workflow_store.create_backlog(body.name)


@router.get("/retraining-backlogs/{backlog_id}")
async def get_retraining_backlog(
    backlog_id: str,
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> dict[str, Any]:
    """Retrieve retraining backlog contents and approval state."""
    backlog = workflow_store.get_backlog(backlog_id)
    if backlog is None:
        raise HTTPException(status_code=404, detail="BACKLOG_UNKNOWN")
    return backlog


@router.post("/retraining-backlogs/{backlog_id}/items", status_code=201)
async def add_retraining_backlog_item(
    backlog_id: str,
    body: RetrainingBacklogItemIn,
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> dict[str, Any]:
    """Add a targeted failure or recipe item to a retraining backlog."""
    try:
        return workflow_store.add_backlog_item(backlog_id, body.failure_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"code": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc


@router.post("/retraining-backlogs/{backlog_id}/approve")
async def approve_retraining_backlog(
    backlog_id: str,
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> dict[str, Any]:
    """Human gate: approve a retraining backlog for downstream training."""
    try:
        return workflow_store.approve_backlog(backlog_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail={"code": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail={"code": str(exc)}) from exc


# ---- Model Comparisons ----


@router.post("/model-comparisons", status_code=201)
async def create_model_comparison(
    body: ModelComparisonIn,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Create a paired model comparison between baseline and candidate runs."""
    del actor_id  # Membership is enforced by the dependency before evidence is loaded.
    baseline = require_scoped_run(store, run_id=body.baseline_run_id, project_id=project_id)
    candidate = require_scoped_run(store, run_id=body.candidate_run_id, project_id=project_id)
    if baseline.get("report") is None or candidate.get("report") is None:
        raise HTTPException(status_code=409, detail="both runs must complete before comparison")

    report = build_defense_report(project_id=project_id, baseline_run=baseline, candidate_run=candidate)
    payload = report.model_dump(mode="json")
    payload.update(body.model_dump(mode="json"))
    payload["paired"] = not bool(report.incompatibilities)
    return store.put_record("model_comparison", report.comparison_id, payload)


@router.post("/comparisons")
async def create_comparison_alias(
    body: ModelComparisonIn,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Compatibility alias for model-comparisons endpoint."""
    return await create_model_comparison(body, project_id=project_id, actor_id=actor_id, store=store)


@router.get("/model-comparisons/{comparison_id}")
async def get_model_comparison(
    comparison_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Retrieve model comparison record."""
    del actor_id
    return require_scoped_record(
        store,
        record_type="model_comparison",
        record_id=comparison_id,
        project_id=project_id,
    )


@router.get("/model-comparisons/{comparison_id}/export")
async def export_model_comparison(
    comparison_id: str,
    format: str = Query(default="json"),
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> Response:
    """Export comparison report in JSON, CSV, or HTML format."""
    del actor_id
    comparison = require_scoped_record(
        store,
        record_type="model_comparison",
        record_id=comparison_id,
        project_id=project_id,
    )
    eligibility = comparison.get("eligibility") or {}
    if eligibility.get("status") != "ELIGIBLE":
        raise HTTPException(
            status_code=409,
            detail={
                "code": "NOT_ELIGIBLE_FOR_CONCLUSION_EXPORT",
                "reasons": eligibility.get("reasons", ["CANONICAL_EVIDENCE_MISSING"]),
            },
        )
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
async def get_model_comparison_metric_deltas(
    comparison_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Retrieve metric deltas for comparison."""
    del actor_id
    comp = require_scoped_record(store, record_type="model_comparison", record_id=comparison_id, project_id=project_id)
    return comp.get("metric_deltas", [])


@router.get("/model-comparisons/{comparison_id}/recovery-report")
async def get_model_comparison_recovery_report(
    comparison_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Retrieve recovery report for comparison."""
    del actor_id
    comp = require_scoped_record(store, record_type="model_comparison", record_id=comparison_id, project_id=project_id)
    return comp.get("recovery", {"reason": "REPORT_MISSING"})


@router.get("/model-comparisons/{comparison_id}/failures")
async def get_model_comparison_failures(
    comparison_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Return failure deltas between baseline and candidate runs."""
    del actor_id
    comp = require_scoped_record(store, record_type="model_comparison", record_id=comparison_id, project_id=project_id)
    return {"comparison_id": comparison_id, **(comp.get("failures") or {})}


# ---- Defence Runs & Candidates ----


@router.get("/defence-checkpoints", response_model=list[ModelVersionOut])
async def list_defence_checkpoints(
    store: SqliteRunStore = Depends(get_store),
) -> list[ModelVersionOut]:
    """List all fine-tuned/repaired defence checkpoints."""
    versions = registered_model_versions(store)
    return [ModelVersionOut.from_domain(v) for v in versions if v.checkpoint_role != "base"]


@router.post("/defence-runs", status_code=202, response_model=RunJobOut)
async def create_defence_run(
    body: DefenceRunIn,
    store: SqliteRunStore = Depends(get_store),
    runner: TestRunner = Depends(get_runner),
    worker: LocalRunWorker = Depends(get_worker),
) -> RunJobOut:
    """Run a reviewed Defence candidate against baseline's locked protocol."""
    baseline = store.get(body.baseline_run_id)
    if baseline is None:
        raise HTTPException(status_code=404, detail={"code": "BASELINE_RUN_UNKNOWN"})
    if baseline.get("status") != "COMPLETED" or baseline.get("report") is None:
        raise HTTPException(status_code=409, detail={"code": "BASELINE_RUN_NOT_COMPLETED"})

    candidate = next((item for item in registered_model_versions(store) if item.id == body.checkpoint_id), None)
    if candidate is None:
        raise HTTPException(status_code=404, detail={"code": "CHECKPOINT_UNKNOWN"})
    if candidate.checkpoint_role == "base":
        raise HTTPException(status_code=422, detail={"code": "DEFENCE_CANDIDATE_ROLE_REQUIRED"})

    raw_config = baseline.get("config") or (baseline.get("report") or {}).get("provenance", {}).get("run_config")
    if not raw_config:
        raise HTTPException(status_code=422, detail={"code": "BASELINE_CONFIG_MISSING"})
    baseline_config = RunConfig.model_validate(raw_config)
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
    config = resolve_run_config(config, allow_defence=True, store=store)
    preflight = runner.preflight(config)
    if preflight.fatal_errors:
        raise HTTPException(status_code=422, detail={"fatal_errors": list(preflight.fatal_errors)})

    run_id = store.create(config)
    store.put_record(
        "defence_evaluation",
        run_id,
        {
            "run_id": run_id,
            "baseline_run_id": body.baseline_run_id,
            "checkpoint_id": candidate.id,
            "protocol_config": baseline_config.model_dump(mode="json"),
        },
    )
    worker.enqueue(run_id, config)
    return job_out(store.get(run_id))


@router.get("/runs/{run_id}/defence-candidates", response_model=list[ModelVersionOut])
async def run_defence_candidates(
    run_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> list[ModelVersionOut]:
    """Retrieve fine-tuned/repaired checkpoints matching the benchmark run task."""
    run = require_run(store, run_id)
    task_id = (run.get("config") or {}).get("task_id", "detection2d")
    return [
        ModelVersionOut.from_domain(version)
        for version in registered_model_versions(store)
        if version.task == task_id and version.checkpoint_role != "base"
    ]


# ---- Model Lineage & Gate Evidence ----


@router.get("/model-versions/{version_id}/lineage", response_model=LineageGraphOut)
async def get_model_lineage(
    version_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> LineageGraphOut:
    versions = registered_model_versions(store)
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
async def get_model_version_benchmark_history(
    version_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Return historical benchmark run reports for a model version."""
    history = []
    for run in store.list():
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
async def get_model_version_gate_evidence(
    version_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Return checkpoint gate outcome and evidence records for a model version."""
    gate = store.get_record("checkpoint_gate", version_id)
    if gate is None:
        for record in store.list_records("checkpoint_gate"):
            if record.get("model_version_id") == version_id or record.get("id") == version_id:
                gate = record
                break
    if gate is None:
        return {"version_id": version_id, "gate_passed": False, "evidence": None, "status": "NO_GATE_EVALUATED"}
    return {"version_id": version_id, "gate_passed": gate.get("passed", False), "evidence": gate}


# ---- Failure Analysis & Failure Clusters ----


@router.get("/failure-cases")
async def list_failure_cases(
    run_id: str | None = Query(default=None),
    store: SqliteRunStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """Aggregate failure cases across completed benchmark runs or query by run_id."""
    failures = []
    runs = [require_run(store, run_id)] if run_id else store.list()
    for run in runs:
        report = run.get("report")
        if report and isinstance(report.get("worst_cases"), list):
            for case in report["worst_cases"]:
                if isinstance(case, dict):
                    failures.append({"run_id": run.get("run_id"), **case})
    return failures


@router.get("/failure-clusters")
async def list_failure_clusters(
    store: SqliteRunStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """List all persisted failure clusters."""
    return store.list_records("failure_cluster")


@router.post("/failure-clusters", status_code=201)
async def create_failure_cluster(
    body: FailureClusterCreateIn,
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Persist a failure cluster grouping related failure cases."""
    cluster_id = body.cluster_id or f"cluster-{uuid.uuid4().hex[:8]}"
    payload = {
        "id": cluster_id,
        "cluster_id": cluster_id,
        "name": body.name,
        "member_ids": body.member_ids,
        "defense_profile_id": body.defense_profile_id,
        "selection_allowed": True,
    }
    return store.put_record("failure_cluster", cluster_id, payload)


@router.get("/failure-clusters/{cluster_id}")
async def get_failure_cluster(
    cluster_id: str,
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Get detail for a specific failure cluster."""
    cluster = store.get_record("failure_cluster", cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"unknown failure cluster {cluster_id!r}")
    return cluster


# ---- Closed-Loop Retraining ----


@router.post("/closed-loop/start", status_code=201, response_model=ClosedLoopSnapshotOut)
async def start_closed_loop(
    body: ClosedLoopStartIn,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> dict[str, Any]:
    """Start a closed-loop retraining pipeline from a completed benchmark run."""
    from src.training.closed_loop import ClosedLoopTracker

    del actor_id
    try:
        item = require_scoped_run(store, run_id=body.run_id, project_id=project_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail={"code": "RUN_UNKNOWN", "run_id": body.run_id}) from exc
        raise
    if item.get("report") is None:
        raise HTTPException(status_code=409, detail={"code": "RUN_NOT_COMPLETED", "run_id": body.run_id})

    reported_cases = item["report"].get("worst_cases", [])
    failures = [failure for failure in reported_cases if isinstance(failure, dict) and is_failure_case(failure)]
    if not failures:
        raise HTTPException(status_code=409, detail={"code": "NO_FAILURE_CASES", "run_id": body.run_id})

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

    loop_id = workflow_store.create_job("closed_loop", {"project_id": project_id, "source_run_id": body.run_id})
    tracker = ClosedLoopTracker(loop_id=loop_id)
    workflow_store.append_event(
        loop_id,
        tracker.state,
        {"source_run_id": body.run_id, "progress_ratio": 0.0},
    )
    job = workflow_store.get_job(loop_id)
    payload = {
        "loop_id": loop_id,
        "project_id": project_id,
        "source_run_id": body.run_id,
        "state": tracker.state,
        "audit": audit,
        "artifacts": {
            "source_benchmark_run": body.run_id,
            "failure_cases": failure_ids,
        },
        "events": workflow_store.events(loop_id),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }
    workflow_store.checkpoint(loop_id, payload)
    return payload


@router.get("/closed-loop/{loop_id}", response_model=ClosedLoopSnapshotOut)
async def get_closed_loop(
    loop_id: str,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> dict[str, Any]:
    """Get current state of a closed-loop pipeline."""
    del actor_id
    try:
        job = require_scoped_workflow_job(workflow_store, job_id=loop_id, project_id=project_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail={"code": "CLOSED_LOOP_UNKNOWN", "loop_id": loop_id}) from exc
        raise
    if job.get("job_type") != "closed_loop":
        raise HTTPException(status_code=404, detail={"code": "CLOSED_LOOP_UNKNOWN", "loop_id": loop_id})
    checkpoints = workflow_store.checkpoints(loop_id)
    if not checkpoints:
        raise HTTPException(status_code=409, detail={"code": "CLOSED_LOOP_SNAPSHOT_MISSING", "loop_id": loop_id})
    return checkpoints[-1]["payload"]


@router.post("/closed-loop/{loop_id}/advance", response_model=ClosedLoopSnapshotOut)
async def advance_closed_loop(
    loop_id: str,
    body: ClosedLoopAdvanceIn,
    project_id: str = Query(..., min_length=1),
    actor_id: str = Depends(require_project_member),
    store: SqliteRunStore = Depends(get_store),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> dict[str, Any]:
    """Advance a recovery loop to a target state if backed by valid persisted evidence."""
    from src.training.closed_loop import ClosedLoopAuditEntry, ClosedLoopTracker, validate_transition

    del actor_id
    try:
        job = require_scoped_workflow_job(workflow_store, job_id=loop_id, project_id=project_id)
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail={"code": "CLOSED_LOOP_UNKNOWN", "loop_id": loop_id}) from exc
        raise
    if job.get("job_type") != "closed_loop":
        raise HTTPException(status_code=404, detail={"code": "CLOSED_LOOP_UNKNOWN", "loop_id": loop_id})
    checkpoints = workflow_store.checkpoints(loop_id)
    if not checkpoints:
        raise HTTPException(status_code=409, detail={"code": "CLOSED_LOOP_SNAPSHOT_MISSING", "loop_id": loop_id})
    latest = checkpoints[-1]["payload"]
    current_state = latest["state"]

    if not validate_transition(current_state, body.target):
        raise HTTPException(
            status_code=409,
            detail={"code": "INVALID_CLOSED_LOOP_TRANSITION", "current": current_state, "target": body.target},
        )

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

    valid_evidence = True
    if body.target == "CLUSTER_FORMED":
        cluster = store.get_record("failure_cluster", body.artifact_id)
        if not cluster:
            valid_evidence = False
    elif body.target in {"BACKLOG_CREATED", "BACKLOG_APPROVED"}:
        backlog = workflow_store.get_backlog(body.artifact_id)
        if not backlog:
            valid_evidence = False
        elif body.target == "BACKLOG_APPROVED" and backlog.get("status") != "APPROVED":
            valid_evidence = False
    elif body.target == "DEFENSE_PROFILED":
        profile = store.get_record("defense_profile", body.artifact_id)
        if not profile:
            valid_evidence = False
    elif body.target == "DATASET_MANIFEST_CREATED":
        manifest = store.get_record("training_dataset_manifest", body.artifact_id)
        if not manifest:
            valid_evidence = False
    elif body.target in {"TRAINING_STARTED", "TRAINING_COMPLETED"}:
        tr_job = workflow_store.get_job(body.artifact_id)
        if not tr_job:
            valid_evidence = False
        elif body.target == "TRAINING_COMPLETED" and tr_job.get("status") != "COMPLETED":
            valid_evidence = False
    elif body.target == "CHECKPOINT_VALIDATED":
        if not body.artifact_id:
            valid_evidence = False
    elif body.target == "GATE_EVALUATED":
        gate = store.get_record("checkpoint_gate", body.artifact_id)
        if not gate:
            valid_evidence = False
    elif body.target == "MODEL_REGISTERED":
        if not body.artifact_id:
            valid_evidence = False
    elif body.target in {"RE_BENCHMARK_STARTED", "RE_BENCHMARK_COMPLETED"}:
        bm_run = store.get(body.artifact_id)
        if not bm_run:
            valid_evidence = False
        elif body.target == "RE_BENCHMARK_COMPLETED" and (not bm_run.get("report")):
            valid_evidence = False
    elif body.target == "RECOVERY_REPORTED":
        comparison = store.get_record("model_comparison", body.artifact_id)
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

    workflow_store.append_event(
        loop_id,
        tracker.state,
        {"artifact_id": body.artifact_id, "artifact_type": art_type},
    )

    audit_payloads = [entry.model_dump(mode="json") for entry in tracker.audit]
    updated_payload = {
        "loop_id": loop_id,
        "project_id": project_id,
        "source_run_id": latest["source_run_id"],
        "state": tracker.state,
        "audit": audit_payloads,
        "artifacts": dict(tracker.artifacts),
        "events": workflow_store.events(loop_id),
        "created_at": latest["created_at"],
        "updated_at": job["updated_at"],
    }
    workflow_store.checkpoint(loop_id, updated_payload)
    return updated_payload
