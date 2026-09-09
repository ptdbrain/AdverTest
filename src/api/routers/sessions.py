"""Durable experiment-session API for the NewUI multi-run workflow.

Render filesystems are ephemeral, so sessions and their run history live in
PostgreSQL rather than ``data/storage/sessions.json``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.platform_dependencies import get_platform_database
from src.persistence.models import ExperimentSessionRecord, ProjectRecord, SessionRunRecord, UserRecord

router = APIRouter(prefix="/sessions", tags=["Experiment Sessions"])

_SYSTEM_USER_ID = "system-demo"
_SYSTEM_PROJECT_ID = "advertest-demo"
_DEFAULT_SESSION_ID = "EXP-2025-0512-001"


class RunRecord(BaseModel):
    id: str
    name: str
    timestamp: str
    attack_type: str
    attack_name: str
    severity: int
    clean_map: float
    attacked_map: float
    map_drop_pct: float
    clean_conf: float
    attacked_conf: float
    psnr: str
    ssim: str
    inference_ms: float
    robustness_score: float
    clean_bbox_count: int
    attacked_bbox_count: int
    sample_id: str = "000000"
    clean_miou: float | None = None
    attacked_miou: float | None = None
    miou_drop_pct: float | None = None
    is_combined: bool = False
    attack_components: list[str] = Field(default_factory=list)
    note: str = ""
    seed: int | None = None
    run_config_hash: str = ""
    backend_run_id: str = ""


class SessionRecord(BaseModel):
    id: str
    name: str
    description: str = ""
    project_id: str | None = None
    task_id: str = "detection2d"
    task_name: str = "Object Detection"
    model_id: str = "local_yolo11s_clean"
    model_name: str = "YOLO11s (Ultralytics Vision)"
    dataset_id: str = "kitti_anonymized_de"
    dataset_name: str = "KITTI Anonymized Set"
    # Workflow §4: {dataset_class: model_class | null | "ignore"} chosen in the
    # mapping matrix; persisted server-side so it survives refreshes.
    class_mapping: dict[str, str | None] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    runs: list[RunRecord] = Field(default_factory=list)
    status: str = "active"
    ended_at: str | None = None
    total_duration_seconds: float | None = None


class UpdateRunNoteIn(BaseModel):
    note: str


def _timestamp(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _decode(payload: str, fallback: object) -> object:
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return fallback


def _ensure_workspace() -> None:
    """Keep browser-created sessions behind actual user/project foreign keys."""
    database = get_platform_database()
    with database.session() as db:
        if db.get(UserRecord, _SYSTEM_USER_ID) is None:
            db.add(UserRecord(
                id=_SYSTEM_USER_ID, email="system@advertest.internal", password_hash="!disabled!",
                display_name="AdverTest system workspace", role="ADMIN", status="ACTIVE",
            ))
    # Commit the identity in its own transaction. This also makes the helper
    # recover cleanly if an earlier project insert was rolled back by a FK
    # violation during an interrupted deployment.
    with database.session() as db:
        if db.get(ProjectRecord, _SYSTEM_PROJECT_ID) is None:
            db.add(ProjectRecord(
                id=_SYSTEM_PROJECT_ID, name="AdverTest default project", owner_user_id=_SYSTEM_USER_ID,
                description="System-owned project for unauthenticated NewUI sessions.",
            ))


def _to_run(row: SessionRunRecord) -> RunRecord:
    return RunRecord(
        id=row.id, name=row.name, timestamp=row.timestamp, attack_type=row.attack_type,
        attack_name=row.attack_name, severity=row.severity, clean_map=row.clean_map,
        attacked_map=row.attacked_map, map_drop_pct=row.map_drop_pct, clean_conf=row.clean_conf,
        attacked_conf=row.attacked_conf, psnr=row.psnr, ssim=row.ssim,
        inference_ms=row.inference_ms, robustness_score=row.robustness_score,
        clean_bbox_count=row.clean_bbox_count, attacked_bbox_count=row.attacked_bbox_count,
        sample_id=row.sample_id, clean_miou=row.clean_miou, attacked_miou=row.attacked_miou,
        miou_drop_pct=row.miou_drop_pct, is_combined=row.is_combined,
        attack_components=_decode(row.attack_components_json, []), note=row.note,
        seed=row.seed, run_config_hash=row.run_config_hash, backend_run_id=row.backend_run_id,
    )


def _to_response(row: ExperimentSessionRecord, run_rows: list[SessionRunRecord]) -> SessionRecord:
    return SessionRecord(
        id=row.id, name=row.name, description=row.description, project_id=row.project_id,
        task_id=row.task_id, task_name=row.task_name, model_id=row.model_id, model_name=row.model_name,
        dataset_id=row.dataset_id, dataset_name=row.dataset_name,
        class_mapping=_decode(row.class_mapping_json, {}),
        created_at=_timestamp(row.created_at) or "", updated_at=_timestamp(row.updated_at) or "",
        ended_at=_timestamp(row.ended_at), status=row.status,
        total_duration_seconds=row.total_duration_seconds,
        runs=[_to_run(item) for item in run_rows],
    )


def _load_response(db, row: ExperimentSessionRecord) -> SessionRecord:
    runs = db.query(SessionRunRecord).filter_by(session_id=row.id).order_by(SessionRunRecord.created_at).all()
    return _to_response(row, runs)


def _apply_run(row: SessionRunRecord, run: RunRecord) -> None:
    row.name, row.timestamp = run.name, run.timestamp
    row.attack_type, row.attack_name, row.severity = run.attack_type, run.attack_name, run.severity
    row.clean_map, row.attacked_map, row.map_drop_pct = run.clean_map, run.attacked_map, run.map_drop_pct
    row.clean_conf, row.attacked_conf = run.clean_conf, run.attacked_conf
    row.psnr, row.ssim, row.inference_ms = run.psnr, run.ssim, run.inference_ms
    row.robustness_score = run.robustness_score
    row.clean_bbox_count, row.attacked_bbox_count = run.clean_bbox_count, run.attacked_bbox_count
    row.sample_id, row.clean_miou, row.attacked_miou, row.miou_drop_pct = run.sample_id, run.clean_miou, run.attacked_miou, run.miou_drop_pct
    row.is_combined, row.attack_components_json = run.is_combined, json.dumps(run.attack_components, ensure_ascii=False)
    row.note, row.seed, row.run_config_hash, row.backend_run_id = run.note, run.seed, run.run_config_hash, run.backend_run_id


def _ensure_default_session() -> None:
    _ensure_workspace()
    with get_platform_database().session() as db:
        if db.get(ExperimentSessionRecord, _DEFAULT_SESSION_ID) is None:
            db.add(ExperimentSessionRecord(
                id=_DEFAULT_SESSION_ID, owner_user_id=_SYSTEM_USER_ID, project_id=_SYSTEM_PROJECT_ID,
                name="Default robustness experiment", description="Persistent default session for the NewUI.",
                task_id="detection2d", task_name="Object Detection (2D)",
                model_id="yolo11n", model_name="YOLO11n (Ultralytics)",
                dataset_id="kitti", dataset_name="KITTI Anonymized Set",
            ))


def _session_or_404(session_id: str, project_id: str | None = None) -> SessionRecord:
    _ensure_workspace()
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None or (project_id and row.project_id != project_id):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        return _load_response(db, row)


@router.get("", response_model=list[SessionRecord])
async def list_sessions(project_id: str | None = Query(default=None)) -> list[SessionRecord]:
    _ensure_default_session()
    with get_platform_database().session() as db:
        query = db.query(ExperimentSessionRecord).order_by(ExperimentSessionRecord.updated_at.desc())
        if project_id:
            query = query.filter(ExperimentSessionRecord.project_id == project_id)
        rows = query.all()
        return [_load_response(db, row) for row in rows]


@router.post("", response_model=SessionRecord)
async def create_or_update_session(session: SessionRecord) -> SessionRecord:
    _ensure_workspace()
    now = datetime.now(UTC)
    with get_platform_database().session() as db:
        # Workflow §3: a session belongs to exactly one project and its task
        # must match the project's task type. Sessions without an explicit
        # project keep attaching to the system demo workspace (legacy flow).
        project_id = session.project_id or _SYSTEM_PROJECT_ID
        project = db.get(ProjectRecord, project_id)
        if project is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
        if session.project_id and project.task_type != session.task_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"TASK_PROJECT_MISMATCH: project '{project.name}' is a "
                    f"{project.task_type} project; session declares '{session.task_id}'."
                ),
            )
        row = db.get(ExperimentSessionRecord, session.id)
        if row is None:
            row = ExperimentSessionRecord(
                id=session.id, owner_user_id=_SYSTEM_USER_ID, project_id=project_id,
                name=session.name, description=session.description, task_id=session.task_id,
                task_name=session.task_name, model_id=session.model_id, model_name=session.model_name,
                dataset_id=session.dataset_id, dataset_name=session.dataset_name,
            )
            db.add(row)
        elif row.project_id != project_id:
            raise HTTPException(status_code=409, detail="SESSION_PROJECT_MISMATCH")
        elif row.status == "completed":
            raise HTTPException(status_code=409, detail="Session has already been completed.")
        row.name, row.description = session.name, session.description
        row.task_id, row.task_name = session.task_id, session.task_name
        row.model_id, row.model_name = session.model_id, session.model_name
        row.dataset_id, row.dataset_name = session.dataset_id, session.dataset_name
        row.class_mapping_json = json.dumps(session.class_mapping, ensure_ascii=False)
        row.status, row.total_duration_seconds, row.updated_at = session.status, session.total_duration_seconds, now
        row.ended_at = now if session.status == "completed" else None
        existing = {item.id: item for item in db.query(SessionRunRecord).filter_by(session_id=row.id).all()}
        submitted = {run.id for run in session.runs}
        for run in session.runs:
            run_row = existing.get(run.id) or SessionRunRecord(id=run.id, session_id=row.id, project_id=row.project_id)
            _apply_run(run_row, run)
            if run.id not in existing:
                db.add(run_row)
        for run_id, run_row in existing.items():
            if run_id not in submitted:
                db.delete(run_row)
        db.flush()
        return _load_response(db, row)


@router.get("/{session_id}", response_model=SessionRecord)
async def get_session(session_id: str, project_id: str | None = Query(default=None)) -> SessionRecord:
    return _session_or_404(session_id, project_id)


@router.post("/{session_id}/runs", response_model=SessionRecord)
async def add_run_to_session(
    session_id: str, run: RunRecord, project_id: str | None = Query(default=None)
) -> SessionRecord:
    _ensure_default_session()
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None:
            row = ExperimentSessionRecord(
                id=session_id, owner_user_id=_SYSTEM_USER_ID, project_id=_SYSTEM_PROJECT_ID,
                name=f"Experiment {session_id}", description="", task_id="detection2d",
                task_name="Object Detection", model_id="yolo11n", model_name="YOLO11n (Ultralytics)",
                dataset_id="kitti", dataset_name="KITTI Anonymized Set",
            )
            db.add(row)
            db.flush()
        if project_id and row.project_id != project_id:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        if row.status == "completed":
            raise HTTPException(status_code=409, detail="Session has already been completed.")
        run_row = db.get(SessionRunRecord, run.id)
        if run_row is None:
            run_row = SessionRunRecord(id=run.id, session_id=row.id, project_id=row.project_id)
            db.add(run_row)
        elif run_row.session_id != row.id:
            raise HTTPException(status_code=409, detail=f"Run '{run.id}' belongs to another session.")
        _apply_run(run_row, run)
        row.updated_at = datetime.now(UTC)
        db.flush()
        return _load_response(db, row)


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    _ensure_workspace()
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is not None:
            db.query(SessionRunRecord).filter_by(session_id=session_id).delete()
            db.delete(row)
    return {"status": "deleted", "session_id": session_id}


@router.delete("/{session_id}/runs/{run_id}", response_model=SessionRecord)
async def delete_run(session_id: str, run_id: str) -> SessionRecord:
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        run = db.query(SessionRunRecord).filter_by(session_id=session_id, id=run_id).first()
        if run is not None:
            db.delete(run)
        row.updated_at = datetime.now(UTC)
        db.flush()
        return _load_response(db, row)


@router.patch("/{session_id}/runs/{run_id}/note", response_model=SessionRecord)
async def update_run_note(
    session_id: str, run_id: str, body: UpdateRunNoteIn, project_id: str | None = Query(default=None)
) -> SessionRecord:
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None or (project_id and row.project_id != project_id):
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        match = db.query(SessionRunRecord).filter_by(session_id=session_id, id=run_id).first()
        if match is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in session.")
        match.note, row.updated_at = body.note, datetime.now(UTC)
        db.flush()
        return _load_response(db, row)


@router.post("/{session_id}/end", response_model=SessionRecord)
async def end_session(session_id: str) -> SessionRecord:
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        if row.status == "completed":
            raise HTTPException(status_code=409, detail="Session has already been completed.")
        now = datetime.now(UTC)
        row.status, row.ended_at, row.updated_at = "completed", now, now
        db.flush()
        return _load_response(db, row)
