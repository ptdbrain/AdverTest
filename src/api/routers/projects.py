"""Authenticated project creation and selection for the product workspace."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import or_

from src.analytics.project_analytics import compute_project_analytics, compute_runs_comparison
from src.api.dependencies import get_store
from src.api.jobs import SqliteRunStore
from src.api.platform_dependencies import get_platform_database
from src.auth.contracts import UserOut
from src.auth.dependencies import get_current_user
from src.persistence.models import ExperimentSessionRecord, ProjectMembershipRecord, ProjectRecord, SessionRunRecord

TaskId = Literal["detection2d", "segmentation", "detection3d"]

router = APIRouter(prefix="/projects", tags=["Projects"])


class CreateProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    # Workflow §3: the task type is fixed for the whole project lifetime.
    task_type: TaskId = "detection2d"


class UpdateProjectIn(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    task_type: TaskId | None = None


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    task_type: str
    owner_user_id: str
    role: str


def _out(project: ProjectRecord, role: str) -> ProjectOut:
    return ProjectOut(
        id=project.id,
        name=project.name,
        description=project.description,
        task_type=project.task_type,
        owner_user_id=project.owner_user_id,
        role=role,
    )


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    user: UserOut = Depends(get_current_user), database=Depends(get_platform_database)
) -> list[ProjectOut]:
    with database.session() as db:
        rows = (
            db.query(ProjectRecord, ProjectMembershipRecord.role)
            .outerjoin(ProjectMembershipRecord, (ProjectMembershipRecord.project_id == ProjectRecord.id) & (ProjectMembershipRecord.user_id == user.id) & (ProjectMembershipRecord.status == "ACTIVE"))
            .filter(or_(ProjectRecord.owner_user_id == user.id, ProjectMembershipRecord.id.isnot(None)))
            .order_by(ProjectRecord.name)
            .all()
        )
        return [_out(project, "OWNER" if project.owner_user_id == user.id else role or "MEMBER") for project, role in rows]


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
async def create_project(
    body: CreateProjectIn, user: UserOut = Depends(get_current_user), database=Depends(get_platform_database)
) -> ProjectOut:
    project = ProjectRecord(
        id=str(uuid4()),
        name=body.name.strip(),
        description=body.description.strip(),
        task_type=body.task_type,
        owner_user_id=user.id,
    )
    with database.session() as db:
        db.add(project)
        db.add(ProjectMembershipRecord(id=str(uuid4()), project_id=project.id, user_id=user.id, role="OWNER", status="ACTIVE"))
    return _out(project, "OWNER")


@router.patch("/{project_id}", response_model=ProjectOut)
async def update_project(
    project_id: str,
    body: UpdateProjectIn,
    user: UserOut = Depends(get_current_user),
    database=Depends(get_platform_database),
) -> ProjectOut:
    with database.session() as db:
        project = _require_visible_project(db, project_id, user)
        if body.task_type is not None and body.task_type != project.task_type:
            # Workflow §3: switching task invalidates every existing session run.
            has_runs = (
                db.query(SessionRunRecord.id)
                .join(ExperimentSessionRecord, SessionRunRecord.session_id == ExperimentSessionRecord.id)
                .filter(ExperimentSessionRecord.project_id == project_id)
                .first()
                is not None
            )
            if has_runs:
                raise HTTPException(status_code=409, detail="TASK_TYPE_LOCKED")
            project.task_type = body.task_type
        if body.name is not None:
            project.name = body.name.strip()
        if body.description is not None:
            project.description = body.description.strip()
        db.flush()
        return _out(project, "OWNER")


def _require_visible_project(db, project_id: str, user: UserOut) -> ProjectRecord:
    project = db.get(ProjectRecord, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    if project.owner_user_id != user.id:
        member = (
            db.query(ProjectMembershipRecord)
            .filter(
                ProjectMembershipRecord.project_id == project_id,
                ProjectMembershipRecord.user_id == user.id,
                ProjectMembershipRecord.status == "ACTIVE",
            )
            .first()
        )
        if member is None:
            raise HTTPException(status_code=403, detail="PROJECT_ACCESS_DENIED")
    return project


def _project_run_items(db, project_id: str) -> list[dict[str, Any]]:
    """Run ids saved into this project's sessions (workflow step 8 links runs to projects)."""
    rows = (
        db.query(SessionRunRecord)
        .join(ExperimentSessionRecord, SessionRunRecord.session_id == ExperimentSessionRecord.id)
        .filter(ExperimentSessionRecord.project_id == project_id)
        .order_by(SessionRunRecord.timestamp)
        .all()
    )
    return [{"run_id": row.id, "name": row.name, "created_at": row.timestamp} for row in rows]


def _completed_report_items(store: SqliteRunStore, metas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for meta in metas:
        stored = store.get(meta["run_id"])
        report = (stored or {}).get("report")
        if isinstance(report, dict):
            items.append({**meta, "report": report})
    return items


@router.get("/{project_id}/analytics")
async def get_project_analytics(
    project_id: str,
    user: UserOut = Depends(get_current_user),
    database=Depends(get_platform_database),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Aggregate every completed run of the project into trend + vulnerability analytics."""
    with database.session() as db:
        _require_visible_project(db, project_id, user)
        metas = _project_run_items(db, project_id)
    return compute_project_analytics(_completed_report_items(store, metas))


@router.get("/{project_id}/runs-comparison")
async def get_project_runs_comparison(
    project_id: str,
    run_ids: str = Query(..., description="Comma-separated run ids, e.g. ?run_ids=a,b,c"),
    user: UserOut = Depends(get_current_user),
    database=Depends(get_platform_database),
    store: SqliteRunStore = Depends(get_store),
) -> dict[str, Any]:
    """Compare N runs of the project: attack × run and severity × run pivots."""
    with database.session() as db:
        _require_visible_project(db, project_id, user)
        metas = _project_run_items(db, project_id)
    requested = list(dict.fromkeys(part.strip() for part in run_ids.split(",") if part.strip()))
    if not requested:
        raise HTTPException(status_code=422, detail="RUN_IDS_REQUIRED")
    known = {meta["run_id"] for meta in metas}
    unknown = [run_id for run_id in requested if run_id not in known]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Runs not saved in this project: {', '.join(unknown)}")
    by_id = {meta["run_id"]: meta for meta in metas}
    ordered = [by_id[run_id] for run_id in requested]
    return compute_runs_comparison(_completed_report_items(store, ordered))
