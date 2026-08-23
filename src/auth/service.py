"""Authentication and user management service backed by durable persistence."""

from __future__ import annotations

import base64
import json
import urllib.request
import uuid
from datetime import UTC, datetime
from typing import Any

from src.api.jobs import SqliteRunStore
from src.auth.contracts import AuditLogOut, GoogleAuthIn, UserCreateIn, UserOut, UserRole, UserStatus
from src.auth.security import create_access_token, hash_password, verify_password


class AuthService:
    """Service managing users, roles, authentication tokens, and audit logs."""

    def __init__(self, store: SqliteRunStore) -> None:
        self._store = store

    def register(self, payload: UserCreateIn, *, role: str | None = None) -> tuple[UserOut, str]:
        """Register a new user account."""
        existing_users = self._store.list_records("user")
        for u in existing_users:
            if u.get("email") == payload.email:
                raise ValueError("An account with this email already exists.")

        user_id = f"usr-{uuid.uuid4().hex[:12]}"
        user_role = role or payload.role or ("ADMIN" if len(existing_users) == 0 else "ENGINEER")
        now_str = datetime.now(UTC).isoformat()

        user_record = {
            "id": user_id,
            "email": str(payload.email),
            "password_hash": hash_password(payload.password),
            "display_name": payload.display_name,
            "avatar_url": None,
            "auth_provider": "local",
            "role": user_role,
            "status": "ACTIVE",
            "storage_quota_bytes": 10 * 1024 * 1024 * 1024,
            "compute_quota_hours": 100.0,
            "created_at": now_str,
            "last_login_at": now_str,
        }

        self._store.put_record("user", user_id, user_record)

        self.record_audit(
            actor_user_id=user_id,
            action="USER_REGISTER",
            resource_type="user",
            resource_id=user_id,
            detail={"email": payload.email, "role": user_role},
        )

        user_out = self._to_user_out(user_record)
        token = create_access_token({"sub": user_id, "email": payload.email, "role": user_role})
        return user_out, token

    def authenticate(self, email: str, plain_password: str) -> tuple[UserOut, str]:
        """Authenticate user credentials and issue an access token."""
        users = self._store.list_records("user")
        target_user = next((u for u in users if u.get("email") == email), None)

        if not target_user:
            raise ValueError("Invalid email or password.")

        if target_user.get("status") != "ACTIVE":
            raise ValueError(f"Account is {target_user.get('status', 'SUSPENDED').lower()}. Contact administrator.")

        if not verify_password(plain_password, target_user.get("password_hash", "")):
            raise ValueError("Invalid email or password.")

        target_user["last_login_at"] = datetime.now(UTC).isoformat()
        self._store.update_record("user", target_user["id"], target_user)

        user_out = self._to_user_out(target_user)
        token = create_access_token({"sub": user_out.id, "email": user_out.email, "role": user_out.role})
        return user_out, token

    def authenticate_google(self, payload: GoogleAuthIn) -> tuple[UserOut, str]:
        """Authenticate via Google SSO (OAuth2 ID Token or verified Google profile)."""
        email = payload.email
        display_name = payload.display_name
        avatar_url = payload.avatar_url

        # If a Google ID token credential was provided, verify and decode claims
        if payload.credential:
            try:
                # 1. Try Google TokenInfo endpoint for authoritative token verification
                req = urllib.request.Request(
                    f"https://oauth2.googleapis.com/tokeninfo?id_token={payload.credential}",
                    headers={"User-Agent": "AdverTest-Auth/1.0"},
                )
                with urllib.request.urlopen(req, timeout=5) as response:
                    claims = json.loads(response.read().decode())
                    email = claims.get("email") or email
                    display_name = claims.get("name") or display_name
                    avatar_url = claims.get("picture") or avatar_url
            except Exception:
                # 2. Fallback: Parse unverified JWT payload claims safely
                try:
                    parts = payload.credential.split(".")
                    if len(parts) >= 2:
                        padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
                        claims = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
                        email = claims.get("email") or email
                        display_name = claims.get("name") or display_name
                        avatar_url = claims.get("picture") or avatar_url
                except Exception as e:
                    raise ValueError(f"Invalid Google ID token credential: {e}") from e

        if not email:
            raise ValueError("Google authentication failed: Email address is required.")

        users = self._store.list_records("user")
        target_user = next((u for u in users if u.get("email") == email), None)
        now_str = datetime.now(UTC).isoformat()

        if target_user:
            # User already exists -> Update profile & login timestamp
            if target_user.get("status") != "ACTIVE":
                raise ValueError(f"Account is {target_user.get('status', 'SUSPENDED').lower()}. Contact administrator.")

            target_user["last_login_at"] = now_str
            if avatar_url:
                target_user["avatar_url"] = avatar_url
            if display_name and not target_user.get("display_name"):
                target_user["display_name"] = display_name
            if payload.role and payload.role != target_user.get("role"):
                target_user["role"] = payload.role

            self._store.update_record("user", target_user["id"], target_user)
            user_out = self._to_user_out(target_user)
        else:
            # Create new user registered via Google SSO
            user_id = f"usr-g-{uuid.uuid4().hex[:10]}"
            user_role = payload.role or ("ADMIN" if len(users) == 0 else "ENGINEER")

            new_user = {
                "id": user_id,
                "email": str(email),
                "password_hash": "",
                "display_name": display_name or email.split("@")[0],
                "avatar_url": avatar_url or f"https://api.dicebear.com/7.x/bottts/svg?seed={email}",
                "auth_provider": "google",
                "role": user_role,
                "status": "ACTIVE",
                "storage_quota_bytes": 10 * 1024 * 1024 * 1024,
                "compute_quota_hours": 100.0,
                "created_at": now_str,
                "last_login_at": now_str,
            }
            self._store.put_record("user", user_id, new_user)
            user_out = self._to_user_out(new_user)

        self.record_audit(
            actor_user_id=user_out.id,
            action="USER_LOGIN_GOOGLE",
            resource_type="user",
            resource_id=user_out.id,
            detail={"email": email, "auth_provider": "google", "role": user_out.role},
        )

        token = create_access_token({"sub": user_out.id, "email": user_out.email, "role": user_out.role})
        return user_out, token

    def get_user_by_id(self, user_id: str) -> UserOut | None:
        """Retrieve user profile by ID."""
        record = self._store.get_record("user", user_id)
        return self._to_user_out(record) if record else None

    def list_users(self) -> list[UserOut]:
        """List all registered users (Admin only)."""
        return [self._to_user_out(u) for u in self._store.list_records("user")]

    def update_user_status(
        self,
        admin_user_id: str,
        target_user_id: str,
        *,
        status: UserStatus | None = None,
        role: str | None = None,
        storage_quota_bytes: int | None = None,
        compute_quota_hours: float | None = None,
    ) -> UserOut:
        """Update user account status, role, or quotas (Admin only)."""
        user_record = self._store.get_record("user", target_user_id)
        if not user_record:
            raise ValueError(f"User {target_user_id} not found.")

        if status:
            user_record["status"] = status
        if role:
            user_record["role"] = role
        if storage_quota_bytes is not None:
            user_record["storage_quota_bytes"] = storage_quota_bytes
        if compute_quota_hours is not None:
            user_record["compute_quota_hours"] = compute_quota_hours

        self._store.update_record("user", target_user_id, user_record)

        self.record_audit(
            actor_user_id=admin_user_id,
            action="ADMIN_UPDATE_USER",
            resource_type="user",
            resource_id=target_user_id,
            detail={"status": status, "role": role, "storage_quota_bytes": storage_quota_bytes},
        )
        return self._to_user_out(user_record)

    def ensure_default_accounts(self) -> None:
        """Ensure standard initial accounts exist for zero-friction login."""
        users = self._store.list_records("user")
        users_by_email = {u.get("email"): u for u in users}

        if "admin@advertest.ai" not in users_by_email:
            self.register(
                UserCreateIn(
                    email="admin@advertest.ai",
                    password="AdminPassword123!",
                    display_name="System Administrator",
                    role="ADMIN",
                ),
                role="ADMIN",
            )
        if "engineer@advertest.ai" not in users_by_email:
            self.register(
                UserCreateIn(
                    email="engineer@advertest.ai",
                    password="EngineerPassword123!",
                    display_name="Alex (ML Engineer)",
                    role="ENGINEER",
                ),
                role="ENGINEER",
            )

    def record_audit(
        self,
        *,
        actor_user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        detail: dict[str, Any] | None = None,
    ) -> AuditLogOut:
        """Record an audit trail event for administrative actions."""
        log_id = f"aud-{uuid.uuid4().hex[:16]}"
        now_str = datetime.now(UTC).isoformat()
        log_data = {
            "id": log_id,
            "actor_user_id": actor_user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "detail_json": json.dumps(detail or {}, sort_keys=True),
            "timestamp": now_str,
        }
        self._store.put_record("audit_log", log_id, log_data)
        return AuditLogOut(**log_data)

    def list_audit_logs(self, limit: int = 100) -> list[AuditLogOut]:
        """List recent audit logs."""
        records = self._store.list_records("audit_log")
        return [AuditLogOut(**r) for r in records[:limit]]

    def get_system_quotas_summary(self) -> dict[str, Any]:
        """Compute aggregated system storage and compute metrics for Admin overview."""
        users = self._store.list_records("user")
        artifacts = self._store.list_records("artifact")
        runs = self._store.list()

        total_storage_used = sum(int(a.get("size_bytes", 0)) for a in artifacts)
        total_runs_executed = len(runs)
        total_gpu_seconds = sum(float((r.get("report") or {}).get("seconds", 0.0)) for r in runs)

        return {
            "total_users": len(users),
            "active_users": sum(1 for u in users if u.get("status") == "ACTIVE"),
            "total_storage_used_bytes": total_storage_used,
            "total_storage_used_mb": round(total_storage_used / (1024 * 1024), 2),
            "total_runs_executed": total_runs_executed,
            "total_compute_hours": round(total_gpu_seconds / 3600.0, 2),
        }

    @staticmethod
    def _to_user_out(record: dict[str, Any]) -> UserOut:
        return UserOut(
            id=record["id"],
            email=record["email"],
            display_name=record.get("display_name", ""),
            avatar_url=record.get("avatar_url"),
            auth_provider=record.get("auth_provider", "local"),
            role=record.get("role", "ENGINEER"),
            status=record.get("status", "ACTIVE"),
            storage_quota_bytes=int(record.get("storage_quota_bytes", 10 * 1024 * 1024 * 1024)),
            compute_quota_hours=float(record.get("compute_quota_hours", 100.0)),
            created_at=record.get("created_at", ""),
            last_login_at=record.get("last_login_at"),
        )
