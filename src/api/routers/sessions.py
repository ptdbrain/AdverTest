"""Durable experiment-session API for the NewUI multi-run workflow.

Render filesystems are ephemeral, so sessions and their run history live in
PostgreSQL rather than ``data/storage/sessions.json``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.platform_dependencies import get_platform_database
from src.persistence.models import ExperimentSessionRecord, ProjectRecord, UserRecord

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
    task_id: str = "detection2d"
    task_name: str = "Object Detection"
    model_id: str = "local_yolo11s_clean"
    model_name: str = "YOLO11s (Ultralytics Vision)"
    dataset_id: str = "kitti_anonymized_de"
    dataset_name: str = "KITTI Anonymized Set"
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
        if db.get(ProjectRecord, _SYSTEM_PROJECT_ID) is None:
            db.add(ProjectRecord(
                id=_SYSTEM_PROJECT_ID, name="AdverTest default project", owner_user_id=_SYSTEM_USER_ID,
                description="System-owned project for unauthenticated NewUI sessions.",
            ))


def _to_response(row: ExperimentSessionRecord) -> SessionRecord:
    metadata = _decode(row.metadata_json, {})
    metadata = metadata if isinstance(metadata, dict) else {}
    run_rows = _decode(row.runs_json, [])
    run_rows = run_rows if isinstance(run_rows, list) else []
    return SessionRecord(
        id=row.id, name=row.name, description=row.description, task_id=row.task_id,
        task_name=str(metadata.get("task_name", "Object Detection")), model_id=row.model_id,
        model_name=str(metadata.get("model_name", row.model_id)), dataset_id=row.dataset_id,
        dataset_name=str(metadata.get("dataset_name", row.dataset_id)),
        created_at=_timestamp(row.created_at) or "", updated_at=_timestamp(row.updated_at) or "",
        ended_at=_timestamp(row.ended_at), status=row.status,
        total_duration_seconds=metadata.get("total_duration_seconds"),
        runs=[RunRecord.model_validate(item) for item in run_rows],
    )


def _ensure_default_session() -> None:
    _ensure_workspace()
    with get_platform_database().session() as db:
        if db.get(ExperimentSessionRecord, _DEFAULT_SESSION_ID) is None:
            db.add(ExperimentSessionRecord(
                id=_DEFAULT_SESSION_ID, user_id=_SYSTEM_USER_ID, project_id=_SYSTEM_PROJECT_ID,
                name="Default robustness experiment", description="Persistent default session for the NewUI.",
                task_id="detection2d", model_id="yolo11n", dataset_id="kitti",
                metadata_json=json.dumps({"task_name": "Object Detection (2D)", "model_name": "YOLO11n (Ultralytics)", "dataset_name": "KITTI Anonymized Set"}),
                runs_json="[]",
            ))


def _session_or_404(session_id: str) -> ExperimentSessionRecord:
    _ensure_workspace()
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        db.expunge(row)
        return row


@router.get("", response_model=list[SessionRecord])
async def list_sessions() -> list[SessionRecord]:
    _ensure_default_session()
    with get_platform_database().session() as db:
        rows = db.query(ExperimentSessionRecord).order_by(ExperimentSessionRecord.updated_at.desc()).all()
        for row in rows:
            db.expunge(row)
    return [_to_response(row) for row in rows]


@router.post("", response_model=SessionRecord)
async def create_or_update_session(session: SessionRecord) -> SessionRecord:
    _ensure_workspace()
    now = datetime.now(UTC)
    metadata = {"task_name": session.task_name, "model_name": session.model_name, "dataset_name": session.dataset_name, "total_duration_seconds": session.total_duration_seconds}
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session.id)
        if row is None:
            row = ExperimentSessionRecord(
                id=session.id, user_id=_SYSTEM_USER_ID, project_id=_SYSTEM_PROJECT_ID,
                name=session.name, description=session.description, task_id=session.task_id,
                model_id=session.model_id, dataset_id=session.dataset_id,
            )
            db.add(row)
        elif row.status == "completed":
            raise HTTPException(status_code=409, detail="Session has already been completed.")
        row.name, row.description = session.name, session.description
        row.task_id, row.model_id, row.dataset_id = session.task_id, session.model_id, session.dataset_id
        row.status, row.metadata_json = session.status, json.dumps(metadata, ensure_ascii=False)
        row.runs_json, row.updated_at = json.dumps([run.model_dump() for run in session.runs], ensure_ascii=False), now
        row.ended_at = now if session.status == "completed" else None
        db.flush()
        db.expunge(row)
    return _to_response(row)


@router.get("/{session_id}", response_model=SessionRecord)
async def get_session(session_id: str) -> SessionRecord:
    return _to_response(_session_or_404(session_id))


@router.post("/{session_id}/runs", response_model=SessionRecord)
async def add_run_to_session(session_id: str, run: RunRecord) -> SessionRecord:
    _ensure_default_session()
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None:
            row = ExperimentSessionRecord(
                id=session_id, user_id=_SYSTEM_USER_ID, project_id=_SYSTEM_PROJECT_ID,
                name=f"Experiment {session_id}", description="", task_id="detection2d",
                model_id="yolo11n", dataset_id="kitti", metadata_json="{}", runs_json="[]",
            )
            db.add(row)
            db.flush()
        if row.status == "completed":
            raise HTTPException(status_code=409, detail="Session has already been completed.")
        runs = _decode(row.runs_json, [])
        if not isinstance(runs, list):
            runs = []
        serialized = run.model_dump()
        index = next((i for i, item in enumerate(runs) if item.get("id") == run.id), None)
        if index is None:
            runs.append(serialized)
        else:
            runs[index] = serialized
        row.runs_json, row.updated_at = json.dumps(runs, ensure_ascii=False), datetime.now(UTC)
        db.flush()
        db.expunge(row)
    return _to_response(row)


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    _ensure_workspace()
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is not None:
            db.delete(row)
    return {"status": "deleted", "session_id": session_id}


@router.delete("/{session_id}/runs/{run_id}", response_model=SessionRecord)
async def delete_run(session_id: str, run_id: str) -> SessionRecord:
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        runs = _decode(row.runs_json, [])
        runs = runs if isinstance(runs, list) else []
        row.runs_json = json.dumps([item for item in runs if item.get("id") != run_id], ensure_ascii=False)
        row.updated_at = datetime.now(UTC)
        db.flush()
        db.expunge(row)
    return _to_response(row)


@router.patch("/{session_id}/runs/{run_id}/note", response_model=SessionRecord)
async def update_run_note(session_id: str, run_id: str, body: UpdateRunNoteIn) -> SessionRecord:
    with get_platform_database().session() as db:
        row = db.get(ExperimentSessionRecord, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        runs = _decode(row.runs_json, [])
        runs = runs if isinstance(runs, list) else []
        match = next((item for item in runs if item.get("id") == run_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in session.")
        match["note"] = body.note
        row.runs_json, row.updated_at = json.dumps(runs, ensure_ascii=False), datetime.now(UTC)
        db.flush()
        db.expunge(row)
    return _to_response(row)


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
        db.expunge(row)
    return _to_response(row)
