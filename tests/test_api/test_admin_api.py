"""Integration tests for Admin API endpoints."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.auth.contracts import UserCreateIn
from src.auth.dependencies import get_auth_service
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_admin_endpoints_require_admin_role(client) -> None:
    auth_service = app.dependency_overrides.get(get_auth_service, get_auth_service)()

    admin_email = f"admin_{uuid.uuid4().hex[:8]}@advertest.ai"
    user_email = f"user_{uuid.uuid4().hex[:8]}@advertest.ai"

    # Register admin
    admin_out, admin_token = auth_service.register(
        UserCreateIn(email=admin_email, password="AdminPassword123!", display_name="Admin"),
        role="ADMIN",
    )
    # Register regular user
    user_out, user_token = auth_service.register(
        UserCreateIn(email=user_email, password="UserPassword123!", display_name="Regular User"),
        role="USER",
    )
    assert admin_out.role == "ADMIN"
    assert user_out.role == "USER"

    # Regular user attempting to access admin endpoint is blocked with 403
    forbidden_res = client.get("/api/v1/admin/users", headers={"Authorization": f"Bearer {user_token}"})
    assert forbidden_res.status_code == 403

    # Admin accessing admin endpoint succeeds
    admin_res = client.get("/api/v1/admin/users", headers={"Authorization": f"Bearer {admin_token}"})
    assert admin_res.status_code == 200
    users = admin_res.json()
    assert len(users) >= 2

    # Admin updating user status
    status_res = client.post(
        f"/api/v1/admin/users/{user_out.id}/status",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"status": "SUSPENDED"},
    )
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "SUSPENDED"

    # Admin checking audit logs
    audit_res = client.get("/api/v1/admin/audit-logs", headers={"Authorization": f"Bearer {admin_token}"})
    assert audit_res.status_code == 200
    assert len(audit_res.json()) > 0
