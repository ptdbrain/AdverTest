"""Admin router: user administration, quotas, system monitoring, and audit log inspection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.auth.contracts import AuditLogOut, UserOut, UserStatusUpdateIn
from src.auth.dependencies import get_auth_service, require_admin
from src.auth.service import AuthService

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


@router.get("/users", response_model=list[UserOut])
async def list_all_users(
    _admin: UserOut = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> list[UserOut]:
    """List all registered users in the system (Admin only)."""
    return auth_service.list_users()


@router.post("/users/{user_id}/status", response_model=UserOut)
async def update_user_status(
    user_id: str,
    payload: UserStatusUpdateIn,
    admin: UserOut = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserOut:
    """Update user account status (ACTIVE, SUSPENDED, DISABLED) or storage/compute quotas (Admin only)."""
    try:
        return auth_service.update_user_status(
            admin_user_id=admin.id,
            target_user_id=user_id,
            status=payload.status,
            role=payload.role,
            storage_quota_bytes=payload.storage_quota_bytes,
            compute_quota_hours=payload.compute_quota_hours,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/quotas")
async def get_system_quotas(
    _admin: UserOut = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """Get aggregate system usage, storage consumption, and active compute metrics (Admin only)."""
    return auth_service.get_system_quotas_summary()


@router.get("/audit-logs", response_model=list[AuditLogOut])
async def get_audit_logs(
    limit: int = Query(default=50, ge=1, le=500),
    _admin: UserOut = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service),
) -> list[AuditLogOut]:
    """Inspect immutable audit logs for administrative compliance (Admin only)."""
    return auth_service.list_audit_logs(limit=limit)
