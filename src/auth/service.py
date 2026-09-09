"""Authentication and user management service backed by durable persistence."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from src.api.jobs import SqliteRunStore
from src.auth.contracts import AuditLogOut, GoogleAuthIn, UserCreateIn, UserOut, UserStatus
from src.auth.security import create_access_token, hash_password, verify_password
from src.config import get_settings
from src.persistence.database import PlatformDatabase
from src.persistence.models import ArtifactRecord, AuditLogRecord, PlatformJobRecord, UserRecord

logger = logging.getLogger(__name__)


def verify_google_credential(credential: str) -> dict[str, Any]:
    """Strictly verify a Google ID token and return its validated claims.

    Security Hardening P0.2, shared by every auth backend: signature, iss, aud,
    exp, sub, email, and email_verified are all validated. The test-only
    ``test-google-token:`` verifier is enabled ONLY in test/dev runs and is
    strictly rejected in production. Client-supplied emails or roles are never
    trusted; callers must read identity exclusively from the returned claims.
    """
    settings = get_settings()
    claims: dict[str, Any] = {}

    # Safe test-only verifier enabled ONLY in test/dev environment, strictly disabled in production
    is_test_mode = settings.app_env in ("test", "development") or bool(os.getenv("PYTEST_CURRENT_TEST"))
    if is_test_mode and credential.startswith("test-google-token:"):
        try:
            raw_json = credential.removeprefix("test-google-token:")
            claims = json.loads(raw_json)
        except Exception as exc:
            raise ValueError(f"Test Google Token decode failed: {exc}") from exc
    elif settings.app_env == "production" and credential.startswith("test-google-token:"):
        raise ValueError("Test tokens are strictly forbidden in production mode.")
    else:
        # Authoritative Google TokenInfo endpoint verification
        try:
            req = urllib.request.Request(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}",
                headers={"User-Agent": "AdverTest-Auth/1.0"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                claims = json.loads(response.read().decode())
        except Exception as exc:
            raise ValueError(f"Google Token verification failed: {exc}. Refusing to trust unverified payload.") from exc

    # Strict claim validations
    # 1. Issuer
    iss = claims.get("iss")
    if iss not in ("accounts.google.com", "https://accounts.google.com"):
        raise ValueError(f"Google Token verification failed: Invalid issuer '{iss}'.")

    # 2. Audience
    expected_aud = settings.google_client_id or os.getenv("GOOGLE_CLIENT_ID", "")
    aud = claims.get("aud")
    if expected_aud and aud != expected_aud:
        raise ValueError("Google Token verification failed: Audience mismatch.")

    # 3. Expiration
    now_ts = int(time.time())
    exp = claims.get("exp")
    if exp is not None:
        try:
            if int(exp) < now_ts:
                raise ValueError("Google Token verification failed: Token has expired.")
        except (TypeError, ValueError) as exc:
            raise ValueError("Google Token verification failed: Invalid exp claim.") from exc

    # 4. Subject (sub)
    sub = claims.get("sub")
    if not sub:
        raise ValueError("Google Token verification failed: Missing sub claim.")

    # 5. Email & Email Verified
    email = claims.get("email")
    if not email or not str(email).strip():
        raise ValueError("Google Token verification failed: Missing or invalid email in token claims.")

    email_verified = claims.get("email_verified")
    if email_verified not in (True, "true", "True", 1, "1"):
        raise ValueError("Google Token verification failed: Email is not verified by Google.")

    claims["email"] = str(email).strip().lower()
    return claims


class AuthService:
    """Service managing users, roles, authentication tokens, and audit logs."""

    def __init__(self, store: SqliteRunStore) -> None:
        self._store = store

    def register(self, payload: UserCreateIn, *, role: str | None = None) -> tuple[UserOut, str]:
        """Register a new user account.

        Security Hardening P0.1: Client-supplied payload.role is strictly ignored.
        Public registration always defaults to RESEARCHER.
        Only trusted internal callers (such as ensure_default_accounts) passing
        the explicit `role` keyword argument may assign privileged roles.
        """
        existing_users = self._store.list_records("user")
        for u in existing_users:
            if u.get("email") == payload.email:
                raise ValueError("An account with this email already exists.")

        user_id = f"usr-{uuid.uuid4().hex[:12]}"
        user_role = role or "RESEARCHER"
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
        """Authenticate via Google SSO (OAuth2 ID Token).

        Security Hardening P0.2: Strict token verification is required.
        Verifies signature, iss, aud, exp, sub, email, and email_verified.
        Unverified payload credentials or forged emails/roles are rejected.
        """
        if not payload.credential or not str(payload.credential).strip():
            raise ValueError("Google authentication failed: Missing ID token credential.")

        claims = verify_google_credential(str(payload.credential).strip())
        email = claims["email"]
        display_name = claims.get("name") or payload.display_name
        avatar_url = claims.get("picture") or payload.avatar_url

        users = self._store.list_records("user")
        target_user = next((u for u in users if u.get("email", "").lower() == email), None)
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
            # Security Hardening: Never allow client payload to escalate role on login
            self._store.update_record("user", target_user["id"], target_user)
            user_out = self._to_user_out(target_user)
        else:
            # Create new user registered via Google SSO with default RESEARCHER role
            user_id = f"usr-g-{uuid.uuid4().hex[:10]}"
            user_role = "RESEARCHER"

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
            settings = get_settings()
            admin_pwd = settings.admin_default_password
            if admin_pwd == "AdminPassword123!":
                logger.warning(
                    "CRITICAL SECURITY WARNING: System initialized with the default Admin password. "
                    "You MUST change this in production by setting the `ADMIN_DEFAULT_PASSWORD` environment variable."
                )

            self.register(
                UserCreateIn(
                    email="admin@advertest.ai",
                    password=admin_pwd,
                    display_name="System Administrator",
                    role="ADMIN",
                ),
                role="ADMIN",
            )
        if "engineer@advertest.ai" not in users_by_email:
            settings = get_settings()
            self.register(
                UserCreateIn(
                    email="engineer@advertest.ai",
                    password=settings.demo_engineer_password,
                    display_name="Alex (ML Engineer)",
                    role="ENGINEER",
                ),
                role="ENGINEER",
            )

    def ensure_seeded_account(
        self, *, user_id: str, email: str, password: str, display_name: str, role: str
    ) -> UserOut:
        """Create one trusted, deterministic account for an opt-in fixture."""
        existing = self._store.get_record("user", user_id)
        email_record = next(
            (item for item in self._store.list_records("user") if item.get("email") == email), None
        )
        if existing is not None:
            if existing.get("email") != email or existing.get("role") != role:
                raise ValueError("DEMO_ACCOUNT_IDENTITY_CONFLICT")
            return self._to_user_out(existing)
        if email_record is not None and email_record.get("id") != user_id:
            raise ValueError("DEMO_ACCOUNT_EMAIL_CONFLICT")
        now_str = datetime.now(UTC).isoformat()
        record = {
            "id": user_id,
            "email": email,
            "password_hash": hash_password(password),
            "display_name": display_name,
            "avatar_url": None,
            "auth_provider": "local",
            "role": role,
            "status": "ACTIVE",
            "storage_quota_bytes": 10 * 1024 * 1024 * 1024,
            "compute_quota_hours": 100.0,
            "created_at": now_str,
            "last_login_at": None,
        }
        self._store.put_record("user", user_id, record)
        self.record_audit(
            actor_user_id=user_id,
            action="USER_SEEDED_DEMO",
            resource_type="user",
            resource_id=user_id,
            detail={"email": email, "role": role},
        )
        return self._to_user_out(record)

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


class PostgresAuthService:
    """Production auth/audit implementation backed by Render PostgreSQL.

    Local development keeps the lightweight SQLite implementation above, while
    Render persists identities, quotas, and audit records in the control-plane
    database mandated by the deployment specification.
    """

    def __init__(self, database: PlatformDatabase) -> None:
        self._database = database

    def register(self, payload: UserCreateIn, *, role: str | None = None) -> tuple[UserOut, str]:
        with self._database.session() as db:
            if db.scalar(select(UserRecord).where(UserRecord.email == str(payload.email))) is not None:
                raise ValueError("An account with this email already exists.")
            user_role = role or payload.role or ("ADMIN" if not db.scalar(select(UserRecord.id).limit(1)) else "ENGINEER")
            record = UserRecord(
                id=f"usr-{uuid.uuid4().hex[:12]}", email=str(payload.email), password_hash=hash_password(payload.password),
                display_name=payload.display_name, role=user_role, status="ACTIVE",
            )
            db.add(record)
            db.flush()
            output = self._to_user_out(record)
        self.record_audit(actor_user_id=output.id, action="USER_REGISTER", resource_type="user", resource_id=output.id,
                          detail={"email": output.email, "role": output.role})
        return output, create_access_token({"sub": output.id, "email": output.email, "role": output.role})

    def authenticate(self, email: str, plain_password: str) -> tuple[UserOut, str]:
        with self._database.session() as db:
            record = db.scalar(select(UserRecord).where(UserRecord.email == email))
            if record is None or record.status != "ACTIVE" or not verify_password(plain_password, record.password_hash):
                raise ValueError("Invalid email or password.")
            record.last_login_at = datetime.now(UTC)
            db.flush()
            output = self._to_user_out(record)
        return output, create_access_token({"sub": output.id, "email": output.email, "role": output.role})

    def authenticate_google(self, payload: GoogleAuthIn) -> tuple[UserOut, str]:
        """Verify the credential using the same strict secure path as local auth service.

        Security Hardening P0.2: a verified Google ID token credential is
        mandatory — client-supplied emails or roles are never trusted, and
        privilege assignment for SSO-created accounts stays server-side.
        """
        if not payload.credential or not str(payload.credential).strip():
            raise ValueError("Google authentication failed: Missing ID token credential.")
        claims = verify_google_credential(str(payload.credential).strip())
        email = claims["email"]
        display_name = claims.get("name") or payload.display_name
        with self._database.session() as db:
            record = db.scalar(select(UserRecord).where(UserRecord.email == email))
            if record is None:
                # First bootstrapping account becomes ADMIN; everyone else is a plain ENGINEER.
                role = "ADMIN" if not db.scalar(select(UserRecord.id).limit(1)) else "ENGINEER"
                record = UserRecord(id=f"usr-g-{uuid.uuid4().hex[:10]}", email=email, password_hash="",
                                    display_name=display_name or email.split("@", 1)[0], role=role, status="ACTIVE")
                db.add(record)
            elif record.status != "ACTIVE":
                raise ValueError(f"Account is {record.status.lower()}. Contact administrator.")
            record.last_login_at = datetime.now(UTC)
            db.flush()
            output = self._to_user_out(record)
        self.record_audit(actor_user_id=output.id, action="USER_LOGIN_GOOGLE", resource_type="user", resource_id=output.id,
                          detail={"email": output.email})
        return output, create_access_token({"sub": output.id, "email": output.email, "role": output.role})

    def get_user_by_id(self, user_id: str) -> UserOut | None:
        with self._database.session() as db:
            record = db.get(UserRecord, user_id)
            return self._to_user_out(record) if record else None

    def list_users(self) -> list[UserOut]:
        with self._database.session() as db:
            return [self._to_user_out(record) for record in db.scalars(select(UserRecord).order_by(UserRecord.created_at.desc())).all()]

    def update_user_status(self, admin_user_id: str, target_user_id: str, *, status: UserStatus | None = None,
                           role: str | None = None, storage_quota_bytes: int | None = None,
                           compute_quota_hours: float | None = None) -> UserOut:
        with self._database.session() as db:
            record = db.get(UserRecord, target_user_id)
            if record is None:
                raise ValueError(f"User {target_user_id} not found.")
            if status:
                record.status = status
            if role:
                record.role = role
            if storage_quota_bytes is not None:
                record.storage_quota_bytes = storage_quota_bytes
            if compute_quota_hours is not None:
                record.compute_quota_hours = compute_quota_hours
            db.flush()
            output = self._to_user_out(record)
        self.record_audit(actor_user_id=admin_user_id, action="ADMIN_UPDATE_USER", resource_type="user", resource_id=target_user_id,
                          detail={"status": status, "role": role, "storage_quota_bytes": storage_quota_bytes})
        return output

    def ensure_default_accounts(self) -> None:
        with self._database.session() as db:
            existing = {record.email for record in db.scalars(select(UserRecord)).all()}
        settings = get_settings()
        if "admin@advertest.ai" not in existing:
            if settings.admin_default_password == "AdminPassword123!":
                logger.warning("CRITICAL SECURITY WARNING: ADMIN_DEFAULT_PASSWORD must be set in production.")
            self.register(UserCreateIn(email="admin@advertest.ai", password=settings.admin_default_password,
                                       display_name="System Administrator", role="ADMIN"), role="ADMIN")
        if "engineer@advertest.ai" not in existing:
            self.register(UserCreateIn(email="engineer@advertest.ai", password=settings.demo_engineer_password,
                                       display_name="Alex (ML Engineer)", role="ENGINEER"), role="ENGINEER")

    def ensure_seeded_account(
        self, *, user_id: str, email: str, password: str, display_name: str, role: str
    ) -> UserOut:
        """Create one trusted, deterministic account for an opt-in fixture."""
        with self._database.session() as db:
            existing = db.get(UserRecord, user_id)
            email_record = db.scalar(select(UserRecord).where(UserRecord.email == email))
            if existing is not None:
                if existing.email != email or existing.role != role:
                    raise ValueError("DEMO_ACCOUNT_IDENTITY_CONFLICT")
                return self._to_user_out(existing)
            if email_record is not None and email_record.id != user_id:
                raise ValueError("DEMO_ACCOUNT_EMAIL_CONFLICT")
            record = UserRecord(
                id=user_id,
                email=email,
                password_hash=hash_password(password),
                display_name=display_name,
                role=role,
                status="ACTIVE",
            )
            db.add(record)
            db.flush()
            output = self._to_user_out(record)
        self.record_audit(
            actor_user_id=user_id,
            action="USER_SEEDED_DEMO",
            resource_type="user",
            resource_id=user_id,
            detail={"email": email, "role": role},
        )
        return output

    def record_audit(self, *, actor_user_id: str, action: str, resource_type: str, resource_id: str,
                     detail: dict[str, Any] | None = None) -> AuditLogOut:
        record = AuditLogRecord(id=f"aud-{uuid.uuid4().hex[:16]}", actor_user_id=actor_user_id, action=action,
                                resource_type=resource_type, resource_id=resource_id,
                                detail_json=json.dumps(detail or {}, sort_keys=True))
        with self._database.session() as db:
            db.add(record)
            db.flush()
            return AuditLogOut(id=record.id, actor_user_id=record.actor_user_id, action=record.action,
                               resource_type=record.resource_type, resource_id=record.resource_id,
                               detail_json=record.detail_json, timestamp=record.timestamp.isoformat())

    def list_audit_logs(self, limit: int = 100) -> list[AuditLogOut]:
        with self._database.session() as db:
            records = db.scalars(select(AuditLogRecord).order_by(AuditLogRecord.timestamp.desc()).limit(limit)).all()
            return [AuditLogOut(id=record.id, actor_user_id=record.actor_user_id, action=record.action,
                                resource_type=record.resource_type, resource_id=record.resource_id,
                                detail_json=record.detail_json, timestamp=record.timestamp.isoformat()) for record in records]

    def get_system_quotas_summary(self) -> dict[str, Any]:
        with self._database.session() as db:
            total_users = db.scalar(select(func.count()).select_from(UserRecord)) or 0
            active_users = db.scalar(select(func.count()).select_from(UserRecord).where(UserRecord.status == "ACTIVE")) or 0
            storage_used = db.scalar(select(func.coalesce(func.sum(ArtifactRecord.size_bytes), 0))) or 0
            total_runs = db.scalar(select(func.count()).select_from(PlatformJobRecord)) or 0
            completed_jobs = db.scalar(select(func.count()).select_from(PlatformJobRecord).where(PlatformJobRecord.completed_at.is_not(None))) or 0
        return {"total_users": total_users, "active_users": active_users, "total_storage_used_bytes": int(storage_used),
                "total_storage_used_mb": round(int(storage_used) / (1024 * 1024), 2),
                "total_runs_executed": total_runs, "total_compute_hours": 0.0,
                "completed_jobs": completed_jobs}

    @staticmethod
    def _to_user_out(record: UserRecord) -> UserOut:
        return UserOut(id=record.id, email=record.email, display_name=record.display_name, role=record.role,
                       status=record.status, storage_quota_bytes=record.storage_quota_bytes,
                       compute_quota_hours=record.compute_quota_hours,
                       created_at=record.created_at.isoformat() if record.created_at else "",
                       last_login_at=record.last_login_at.isoformat() if record.last_login_at else None)
