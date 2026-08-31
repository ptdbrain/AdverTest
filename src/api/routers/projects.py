"""Authenticated project creation and selection for the product workspace."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy import or_

from src.api.platform_dependencies import get_platform_database
from src.auth.contracts import UserOut
from src.auth.dependencies import get_current_user
from src.persistence.models import ProjectMembershipRecord, ProjectRecord

router = APIRouter(prefix="/projects", tags=["Projects"])


class CreateProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    owner_user_id: str
    role: str


def _out(project: ProjectRecord, role: str) -> ProjectOut:
    return ProjectOut(id=project.id, name=project.name, description=project.description, owner_user_id=project.owner_user_id, role=role)


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
    project = ProjectRecord(id=str(uuid4()), name=body.name.strip(), description=body.description.strip(), owner_user_id=user.id)
    with database.session() as db:
        db.add(project)
        db.add(ProjectMembershipRecord(id=str(uuid4()), project_id=project.id, user_id=user.id, role="OWNER", status="ACTIVE"))
    return _out(project, "OWNER")
