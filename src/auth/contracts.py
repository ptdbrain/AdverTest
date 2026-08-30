"""Contracts and data models for Authentication, User Management, and Admin RBAC."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

UserRole = Literal["RESEARCHER", "USER", "ADMIN", "ENGINEER", "REVIEWER", "AUDITOR"]
UserStatus = Literal["ACTIVE", "SUSPENDED", "DISABLED"]


class UserOut(BaseModel):
    """Public user profile."""

    id: str
    email: str
    display_name: str
    avatar_url: str | None = None
    auth_provider: str = "local"
    role: str = "ENGINEER"
    status: UserStatus = "ACTIVE"
    storage_quota_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB default
    compute_quota_hours: float = 100.0  # 100 GPU/CPU hours default
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_login_at: str | None = None


class UserCreateIn(BaseModel):
    """Payload for registering a new user."""

    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=8, description="Password must have at least 8 characters")
    display_name: str = Field(min_length=1, max_length=100)
    role: str | None = None


class LoginIn(BaseModel):
    """Payload for authenticating user credentials."""

    email: str
    password: str


class GoogleAuthIn(BaseModel):
    """Payload for authenticating with Google SSO (ID Token or OAuth callback)."""

    model_config = ConfigDict(extra="forbid")

    credential: str = Field(min_length=1)


class TokenOut(BaseModel):
    """Access token payload returned upon successful authentication."""

    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int
    user: UserOut


class UserStatusUpdateIn(BaseModel):
    """Admin payload for updating user account status, role, or quotas."""

    status: UserStatus | None = None
    role: str | None = None
    storage_quota_bytes: int | None = None
    compute_quota_hours: float | None = None


class AuditLogOut(BaseModel):
    """Audit log entry for sensitive operations and admin actions."""

    id: str
    actor_user_id: str
    action: str
    resource_type: str
    resource_id: str
    detail_json: str = "{}"
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
