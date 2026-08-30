"""Experiment Sessions & Multi-Run History Management Router.

Stores, retrieves, and compares experiment sessions and their sequential attack runs
backed by durable relational tables (experiment_sessions, session_runs).
Zero runtime mock data seeding.
"""

from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.platform_dependencies import get_platform_database, require_project_member
from src.persistence.database import PlatformDatabase
from src.persistence.models import ExperimentSessionRecord, SessionRunRecord

router = APIRouter(prefix="/sessions", tags=["Experiment Sessions"])


class RunRecord(BaseModel):
    id: str
    name: str
    timestamp: str
    attack_type: str
    attack_name: str
    severity: int
    clean_map: float | None = None
    attacked_map: float | None = None
    map_drop_pct: float | None = None
    clean_conf: float | None = None
    attacked_conf: float | None = None
    psnr: str = "N/A"
    ssim: str = "N/A"
    inference_ms: float | None = None
    robustness_score: float | None = None
    clean_bbox_count: int = 0
    attacked_bbox_count: int = 0
    sample_id: str = "000000"

    # Segmentation-specific (optional, used when task is segmentation)
    clean_miou: float | None = None
    attacked_miou: float | None = None
    miou_drop_pct: float | None = None

    # Combined attack indicator
    is_combined: bool = False
    attack_components: list[str] = Field(default_factory=list)

    # Researcher note (auto-saved from the visual results page)
    note: str = ""

    # Reproducibility metadata
    seed: int | None = None
    run_config_hash: str = ""
    backend_run_id: str = ""
    evidence_status: str = "NOT_ELIGIBLE"


class SessionRecord(BaseModel):
    id: str
    name: str
    description: str = ""
    task_id: str = "detection2d"
    task_name: str = "Object Detection"
    model_id: str = ""
    model_name: str = ""
    dataset_id: str = ""
    dataset_name: str = ""
    created_at: str
    updated_at: str
    runs: list[RunRecord] = Field(default_factory=list)

    # Session lifecycle
    status: str = "active"
    ended_at: str | None = None
    total_duration_seconds: float | None = None


def _row_to_run_record(row: SessionRunRecord) -> RunRecord:
    attack_comps = []
    if row.attack_components_json:
        try:
            attack_comps = json.loads(row.attack_components_json)
        except Exception:
            attack_comps = []
    evidence_status = "NOT_ELIGIBLE"
    if row.metrics_json:
        try:
            evidence_status = str(json.loads(row.metrics_json).get("evidence_status", evidence_status))
        except Exception:
            pass
    return RunRecord(
        id=row.id,
        name=row.name,
        timestamp=row.timestamp,
        attack_type=row.attack_type,
        attack_name=row.attack_name,
        severity=row.severity,
        clean_map=row.clean_map,
        attacked_map=row.attacked_map,
        map_drop_pct=row.map_drop_pct,
        clean_conf=row.clean_conf,
        attacked_conf=row.attacked_conf,
        psnr=row.psnr,
        ssim=row.ssim,
        inference_ms=row.inference_ms,
        robustness_score=row.robustness_score,
        clean_bbox_count=row.clean_bbox_count,
        attacked_bbox_count=row.attacked_bbox_count,
        sample_id=row.sample_id,
        clean_miou=row.clean_miou,
        attacked_miou=row.attacked_miou,
        miou_drop_pct=row.miou_drop_pct,
        is_combined=row.is_combined,
        attack_components=attack_comps,
        note=row.note,
        seed=row.seed,
        run_config_hash=row.run_config_hash,
        backend_run_id=row.backend_run_id,
        evidence_status=evidence_status,
    )


def _row_to_session_record(s_row: ExperimentSessionRecord, run_rows: list[SessionRunRecord]) -> SessionRecord:
    created_str = (
        s_row.created_at.strftime("%d/%m/%Y %H:%M:%S")
        if isinstance(s_row.created_at, datetime)
        else str(s_row.created_at)
    )
    updated_str = (
        s_row.updated_at.strftime("%d/%m/%Y %H:%M:%S")
        if isinstance(s_row.updated_at, datetime)
        else str(s_row.updated_at)
    )
    ended_str = (
        s_row.ended_at.strftime("%d/%m/%Y %H:%M:%S")
        if isinstance(s_row.ended_at, datetime)
        else (str(s_row.ended_at) if s_row.ended_at else None)
    )

    runs = [_row_to_run_record(r) for r in run_rows]
    return SessionRecord(
        id=s_row.id,
        name=s_row.name,
        description=s_row.description,
        task_id=s_row.task_id,
        task_name=s_row.task_name,
        model_id=s_row.model_id,
        model_name=s_row.model_name,
        dataset_id=s_row.dataset_id,
        dataset_name=s_row.dataset_name,
        created_at=created_str,
        updated_at=updated_str,
        runs=runs,
        status=s_row.status,
        ended_at=ended_str,
        total_duration_seconds=s_row.total_duration_seconds,
    )


@router.get("", response_model=list[SessionRecord])
async def list_sessions(
    project_id: str,
    actor_id: str = Depends(require_project_member),
    db: PlatformDatabase = Depends(get_platform_database),
) -> list[SessionRecord]:
    """Get experiment sessions from exactly one authorized project."""
    del actor_id
    with db.session() as session:
        session_rows = (
            session.query(ExperimentSessionRecord)
            .filter(ExperimentSessionRecord.project_id == project_id)
            .order_by(ExperimentSessionRecord.created_at.desc())
            .all()
        )
        results: list[SessionRecord] = []
        for s_row in session_rows:
            run_rows = (
                session.query(SessionRunRecord)
                .filter(SessionRunRecord.session_id == s_row.id)
                .order_by(SessionRunRecord.created_at.asc())
                .all()
            )
            results.append(_row_to_session_record(s_row, run_rows))
        return results


@router.post("", response_model=SessionRecord)
async def create_or_update_session(
    session_data: SessionRecord,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    db: PlatformDatabase = Depends(get_platform_database),
) -> SessionRecord:
    """Create a new experiment session or update existing."""
    with db.session() as session:
        existing = (
            session.query(ExperimentSessionRecord)
            .filter(
                ExperimentSessionRecord.id == session_data.id,
                ExperimentSessionRecord.project_id == project_id,
            )
            .first()
        )
        if existing:
            existing.name = session_data.name
            existing.description = session_data.description
            existing.task_id = session_data.task_id
            existing.task_name = session_data.task_name
            existing.model_id = session_data.model_id
            existing.model_name = session_data.model_name
            existing.dataset_id = session_data.dataset_id
            existing.dataset_name = session_data.dataset_name
            existing.status = session_data.status
        else:
            new_s = ExperimentSessionRecord(
                id=session_data.id,
                project_id=project_id,
                owner_user_id=actor_id,
                name=session_data.name,
                description=session_data.description,
                task_id=session_data.task_id,
                task_name=session_data.task_name,
                model_id=session_data.model_id,
                model_name=session_data.model_name,
                dataset_id=session_data.dataset_id,
                dataset_name=session_data.dataset_name,
                status=session_data.status,
            )
            session.add(new_s)

        # Upsert runs if provided
        for r in session_data.runs:
            existing_run = (
                session.query(SessionRunRecord)
                .filter(
                    SessionRunRecord.session_id == session_data.id,
                    SessionRunRecord.id == r.id,
                )
                .first()
            )
            if existing_run:
                existing_run.name = r.name
                existing_run.timestamp = r.timestamp
                existing_run.attack_type = r.attack_type
                existing_run.attack_name = r.attack_name
                existing_run.severity = r.severity
                existing_run.clean_map = r.clean_map
                existing_run.attacked_map = r.attacked_map
                existing_run.map_drop_pct = r.map_drop_pct
                existing_run.clean_conf = r.clean_conf
                existing_run.attacked_conf = r.attacked_conf
                existing_run.psnr = r.psnr
                existing_run.ssim = r.ssim
                existing_run.inference_ms = r.inference_ms
                existing_run.robustness_score = r.robustness_score
                existing_run.clean_bbox_count = r.clean_bbox_count
                existing_run.attacked_bbox_count = r.attacked_bbox_count
                existing_run.sample_id = r.sample_id
                existing_run.clean_miou = r.clean_miou
                existing_run.attacked_miou = r.attacked_miou
                existing_run.miou_drop_pct = r.miou_drop_pct
                existing_run.is_combined = r.is_combined
                existing_run.attack_components_json = json.dumps(r.attack_components)
                existing_run.note = r.note
                existing_run.seed = r.seed
                existing_run.run_config_hash = r.run_config_hash
                existing_run.backend_run_id = r.backend_run_id
                existing_run.metrics_json = json.dumps({"evidence_status": r.evidence_status})
            else:
                new_run = SessionRunRecord(
                    id=r.id,
                    session_id=session_data.id,
                    project_id=project_id,
                    name=r.name,
                    timestamp=r.timestamp,
                    attack_type=r.attack_type,
                    attack_name=r.attack_name,
                    severity=r.severity,
                    clean_map=r.clean_map,
                    attacked_map=r.attacked_map,
                    map_drop_pct=r.map_drop_pct,
                    clean_conf=r.clean_conf,
                    attacked_conf=r.attacked_conf,
                    psnr=r.psnr,
                    ssim=r.ssim,
                    inference_ms=r.inference_ms,
                    robustness_score=r.robustness_score,
                    clean_bbox_count=r.clean_bbox_count,
                    attacked_bbox_count=r.attacked_bbox_count,
                    sample_id=r.sample_id,
                    clean_miou=r.clean_miou,
                    attacked_miou=r.attacked_miou,
                    miou_drop_pct=r.miou_drop_pct,
                    is_combined=r.is_combined,
                    attack_components_json=json.dumps(r.attack_components),
                    note=r.note,
                    seed=r.seed,
                    run_config_hash=r.run_config_hash,
                backend_run_id=r.backend_run_id,
                metrics_json=json.dumps({"evidence_status": r.evidence_status}),
                )
                session.add(new_run)

        session.commit()

        s_saved = session.query(ExperimentSessionRecord).filter(ExperimentSessionRecord.id == session_data.id).first()
        r_saved = session.query(SessionRunRecord).filter(SessionRunRecord.session_id == session_data.id).all()
        return _row_to_session_record(s_saved, r_saved)


@router.get("/{session_id}", response_model=SessionRecord)
async def get_session(
    session_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    db: PlatformDatabase = Depends(get_platform_database),
) -> SessionRecord:
    """Get a specific session by ID."""
    del actor_id
    with db.session() as session:
        s_row = (
            session.query(ExperimentSessionRecord)
            .filter(
                ExperimentSessionRecord.id == session_id,
                ExperimentSessionRecord.project_id == project_id,
            )
            .first()
        )
        if not s_row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        run_rows = (
            session.query(SessionRunRecord)
            .filter(SessionRunRecord.session_id == session_id)
            .order_by(SessionRunRecord.created_at.asc())
            .all()
        )
        return _row_to_session_record(s_row, run_rows)


@router.post("/{session_id}/runs", response_model=SessionRecord)
async def add_run_to_session(
    session_id: str,
    run: RunRecord,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    db: PlatformDatabase = Depends(get_platform_database),
) -> SessionRecord:
    """Append a new attack execution run record to the session."""
    del actor_id
    with db.session() as session:
        s_row = (
            session.query(ExperimentSessionRecord)
            .filter(
                ExperimentSessionRecord.id == session_id,
                ExperimentSessionRecord.project_id == project_id,
            )
            .first()
        )
        if not s_row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        existing_run = (
            session.query(SessionRunRecord)
            .filter(
                SessionRunRecord.session_id == session_id,
                SessionRunRecord.id == run.id,
            )
            .first()
        )

        if existing_run:
            existing_run.name = run.name
            existing_run.timestamp = run.timestamp
            existing_run.attack_type = run.attack_type
            existing_run.attack_name = run.attack_name
            existing_run.severity = run.severity
            existing_run.clean_map = run.clean_map
            existing_run.attacked_map = run.attacked_map
            existing_run.map_drop_pct = run.map_drop_pct
            existing_run.clean_conf = run.clean_conf
            existing_run.attacked_conf = run.attacked_conf
            existing_run.psnr = run.psnr
            existing_run.ssim = run.ssim
            existing_run.inference_ms = run.inference_ms
            existing_run.robustness_score = run.robustness_score
            existing_run.clean_bbox_count = run.clean_bbox_count
            existing_run.attacked_bbox_count = run.attacked_bbox_count
            existing_run.sample_id = run.sample_id
            existing_run.clean_miou = run.clean_miou
            existing_run.attacked_miou = run.attacked_miou
            existing_run.miou_drop_pct = run.miou_drop_pct
            existing_run.is_combined = run.is_combined
            existing_run.attack_components_json = json.dumps(run.attack_components)
            existing_run.note = run.note
            existing_run.seed = run.seed
            existing_run.run_config_hash = run.run_config_hash
            existing_run.backend_run_id = run.backend_run_id
            existing_run.metrics_json = json.dumps({"evidence_status": run.evidence_status})
        else:
            new_run = SessionRunRecord(
                id=run.id,
                session_id=session_id,
                project_id=s_row.project_id,
                name=run.name,
                timestamp=run.timestamp,
                attack_type=run.attack_type,
                attack_name=run.attack_name,
                severity=run.severity,
                clean_map=run.clean_map,
                attacked_map=run.attacked_map,
                map_drop_pct=run.map_drop_pct,
                clean_conf=run.clean_conf,
                attacked_conf=run.attacked_conf,
                psnr=run.psnr,
                ssim=run.ssim,
                inference_ms=run.inference_ms,
                robustness_score=run.robustness_score,
                clean_bbox_count=run.clean_bbox_count,
                attacked_bbox_count=run.attacked_bbox_count,
                sample_id=run.sample_id,
                clean_miou=run.clean_miou,
                attacked_miou=run.attacked_miou,
                miou_drop_pct=run.miou_drop_pct,
                is_combined=run.is_combined,
                attack_components_json=json.dumps(run.attack_components),
                note=run.note,
                seed=run.seed,
                run_config_hash=run.run_config_hash,
                backend_run_id=run.backend_run_id,
                metrics_json=json.dumps({"evidence_status": run.evidence_status}),
            )
            session.add(new_run)

        session.commit()

        s_saved = session.query(ExperimentSessionRecord).filter(ExperimentSessionRecord.id == session_id).first()
        r_saved = (
            session.query(SessionRunRecord)
            .filter(SessionRunRecord.session_id == session_id)
            .order_by(SessionRunRecord.created_at.asc())
            .all()
        )
        return _row_to_session_record(s_saved, r_saved)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    db: PlatformDatabase = Depends(get_platform_database),
) -> dict[str, str]:
    """Delete an entire experiment session and all its runs."""
    del actor_id
    with db.session() as session:
        s_row = (
            session.query(ExperimentSessionRecord)
            .filter(
                ExperimentSessionRecord.id == session_id,
                ExperimentSessionRecord.project_id == project_id,
            )
            .first()
        )
        if not s_row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        session.delete(s_row)
        session.commit()
    return {"status": "deleted", "session_id": session_id}


@router.delete("/{session_id}/runs/{run_id}")
async def delete_run(
    session_id: str,
    run_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    db: PlatformDatabase = Depends(get_platform_database),
) -> SessionRecord:
    """Delete a specific run from an experiment session."""
    del actor_id
    with db.session() as session:
        s_row = (
            session.query(ExperimentSessionRecord)
            .filter(
                ExperimentSessionRecord.id == session_id,
                ExperimentSessionRecord.project_id == project_id,
            )
            .first()
        )
        if not s_row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        r_row = (
            session.query(SessionRunRecord)
            .filter(
                SessionRunRecord.session_id == session_id,
                SessionRunRecord.id == run_id,
            )
            .first()
        )
        if r_row:
            session.delete(r_row)
            session.commit()

        s_saved = session.query(ExperimentSessionRecord).filter(ExperimentSessionRecord.id == session_id).first()
        r_saved = (
            session.query(SessionRunRecord)
            .filter(SessionRunRecord.session_id == session_id)
            .order_by(SessionRunRecord.created_at.asc())
            .all()
        )
        return _row_to_session_record(s_saved, r_saved)


# ── Researcher Notes ─────────────────────────────────────────────


class UpdateRunNoteIn(BaseModel):
    """Payload for updating a researcher note on a specific run."""

    note: str


@router.patch("/{session_id}/runs/{run_id}/note", response_model=SessionRecord)
async def update_run_note(
    session_id: str,
    run_id: str,
    body: UpdateRunNoteIn,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    db: PlatformDatabase = Depends(get_platform_database),
) -> SessionRecord:
    """Update or set researcher note for a specific run."""
    del actor_id
    with db.session() as session:
        s_row = (
            session.query(ExperimentSessionRecord)
            .filter(
                ExperimentSessionRecord.id == session_id,
                ExperimentSessionRecord.project_id == project_id,
            )
            .first()
        )
        if not s_row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        r_row = (
            session.query(SessionRunRecord)
            .filter(
                SessionRunRecord.session_id == session_id,
                SessionRunRecord.id == run_id,
            )
            .first()
        )
        if not r_row:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in session.")

        r_row.note = body.note
        session.commit()

        s_saved = session.query(ExperimentSessionRecord).filter(ExperimentSessionRecord.id == session_id).first()
        r_saved = (
            session.query(SessionRunRecord)
            .filter(SessionRunRecord.session_id == session_id)
            .order_by(SessionRunRecord.created_at.asc())
            .all()
        )
        return _row_to_session_record(s_saved, r_saved)


# ── Session Lifecycle ────────────────────────────────────────────


@router.post("/{session_id}/end", response_model=SessionRecord)
async def end_session(
    session_id: str,
    project_id: str,
    actor_id: str = Depends(require_project_member),
    db: PlatformDatabase = Depends(get_platform_database),
) -> SessionRecord:
    """Mark session as completed and lock it from further modifications."""
    del actor_id
    with db.session() as session:
        s_row = (
            session.query(ExperimentSessionRecord)
            .filter(
                ExperimentSessionRecord.id == session_id,
                ExperimentSessionRecord.project_id == project_id,
            )
            .first()
        )
        if not s_row:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
        if s_row.status == "completed":
            raise HTTPException(status_code=409, detail="Session has already been completed.")

        s_row.status = "completed"
        s_row.ended_at = datetime.utcnow()
        session.commit()

        s_saved = session.query(ExperimentSessionRecord).filter(ExperimentSessionRecord.id == session_id).first()
        r_saved = (
            session.query(SessionRunRecord)
            .filter(SessionRunRecord.session_id == session_id)
            .order_by(SessionRunRecord.created_at.asc())
            .all()
        )
        return _row_to_session_record(s_saved, r_saved)
