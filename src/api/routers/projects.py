"""Authenticated project bootstrap and membership listing endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from src.api.platform_dependencies import get_platform_database
from src.auth.contracts import UserOut
from src.auth.dependencies import get_current_user
from src.persistence.database import PlatformDatabase
from src.persistence.models import ProjectMembershipRecord, ProjectRecord, UserRecord

router = APIRouter(prefix="/projects", tags=["Projects"])


class CreateProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=10_000)


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str
    membership_role: str
    status: str


def _ensure_relational_user(db_session, user: UserOut) -> None:
    """Maintain the FK bridge while legacy credential records are migrated."""
    if db_session.get(UserRecord, user.id) is None:
        db_session.add(
            UserRecord(
                id=user.id,
                email=user.email,
                password_hash="managed-by-auth-service",
                display_name=user.display_name,
                role=user.role,
                status=user.status,
                storage_quota_bytes=user.storage_quota_bytes,
                compute_quota_hours=user.compute_quota_hours,
            )
        )


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    current_user: UserOut = Depends(get_current_user),
    db: PlatformDatabase = Depends(get_platform_database),
) -> list[ProjectOut]:
    with db.session() as session:
        memberships = (
            session.query(ProjectMembershipRecord, ProjectRecord)
            .join(ProjectRecord, ProjectRecord.id == ProjectMembershipRecord.project_id)
            .filter(
                ProjectMembershipRecord.user_id == current_user.id,
                ProjectMembershipRecord.status == "ACTIVE",
            )
            .order_by(ProjectRecord.created_at.desc())
            .all()
        )
        return [
            ProjectOut(
                id=project.id,
                name=project.name,
                description=project.description,
                membership_role=membership.role,
                status=membership.status,
            )
            for membership, project in memberships
        ]


@router.post("", response_model=ProjectOut, status_code=status.HTTP_201_CREATED)
async def create_project(
    payload: CreateProjectIn,
    current_user: UserOut = Depends(get_current_user),
    db: PlatformDatabase = Depends(get_platform_database),
) -> ProjectOut:
    project_id = f"project-{uuid.uuid4().hex}"
    with db.session() as session:
        _ensure_relational_user(session, current_user)
        session.add(
            ProjectRecord(
                id=project_id,
                name=payload.name.strip(),
                description=payload.description,
                owner_user_id=current_user.id,
            )
        )
        session.add(
            ProjectMembershipRecord(
                id=f"membership-{uuid.uuid4().hex}",
                project_id=project_id,
                user_id=current_user.id,
                role="OWNER",
                status="ACTIVE",
            )
        )
    return ProjectOut(
        id=project_id,
        name=payload.name.strip(),
        description=payload.description,
        membership_role="OWNER",
        status="ACTIVE",
    )
