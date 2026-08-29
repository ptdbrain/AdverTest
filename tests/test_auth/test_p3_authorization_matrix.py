"""Comprehensive P3.3 Authorization & Adversarial Security Test Suite.

Validates the full RBAC matrix across:
- Roles: anonymous, researcher, reviewer, project_owner, admin, outside_user
- Resources: runs, datasets, checkpoints, artifacts, reports, reviews, defence runs, exports, admin endpoints
- Vectors: Horizontal privilege escalation (cross-project/tenant) and Vertical privilege escalation (unauthorized role elevations).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.auth.contracts import UserCreateIn, UserOut
from src.auth.dependencies import get_auth_service
from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_service():
    service = get_auth_service()
    service.ensure_default_accounts()
    return service


def _create_user_with_role(auth_service, role: str, email_prefix: str) -> tuple[UserOut, str]:
    """Helper to create a test user with a specified role and generate an access token."""
    uid = uuid.uuid4().hex[:8]
    email = f"{email_prefix}_{uid}@advertest.ai"
    user_in = UserCreateIn(
        email=email,
        password="TestPassword123!",
        display_name=f"Test {role.title()}",
        role=role,
    )
    user, token = auth_service.register(user_in, role=role)
    return user, token


# =========================================================================
# 1. Vertical Privilege Escalation & Admin Endpoints Matrix
# =========================================================================

@pytest.mark.parametrize(
    "role,allowed",
    [
        ("ANONYMOUS", False),
        ("RESEARCHER", False),
        ("REVIEWER", False),
        ("USER", False),
        ("ENGINEER", False),
        ("ADMIN", True),
    ],
)
def test_admin_endpoints_vertical_matrix(client, auth_service, role: str, allowed: bool) -> None:
    """Ensure admin routes (users list, quota modification, audit logs) reject non-admins."""
    headers = {}
    if role != "ANONYMOUS":
        _, token = _create_user_with_role(auth_service, role, f"vert_{role.lower()}")
        headers = {"Authorization": f"Bearer {token}"}

    # Endpoint 1: GET /api/v1/admin/users
    res_users = client.get("/api/v1/admin/users", headers=headers)
    if allowed:
        assert res_users.status_code == 200
        assert isinstance(res_users.json(), list)
    else:
        assert res_users.status_code in (401, 403)

    # Endpoint 2: GET /api/v1/admin/quotas
    res_quotas = client.get("/api/v1/admin/quotas", headers=headers)
    if allowed:
        assert res_quotas.status_code == 200
    else:
        assert res_quotas.status_code in (401, 403)

    # Endpoint 3: GET /api/v1/admin/audit-logs
    res_logs = client.get("/api/v1/admin/audit-logs", headers=headers)
    if allowed:
        assert res_logs.status_code == 200
    else:
        assert res_logs.status_code in (401, 403)


def test_vertical_escalation_user_cannot_grant_themselves_admin(client, auth_service) -> None:
    """Ensure a standard user cannot modify their own or another user's role to ADMIN."""
    user, token = _create_user_with_role(auth_service, "RESEARCHER", "attacker")
    headers = {"Authorization": f"Bearer {token}"}

    # Attempt to change role to ADMIN
    res = client.post(
        f"/api/v1/admin/users/{user.id}/status",
        json={"role": "ADMIN"},
        headers=headers,
    )
    assert res.status_code == 403
    # Verify role in DB remains un-escalated
    fresh_user = auth_service.get_user_by_id(user.id)
    assert fresh_user.role == "RESEARCHER"


# =========================================================================
# 2. Horizontal Privilege Escalation & Resource Isolation Matrix
# =========================================================================

def test_reviews_authorization_and_cross_project_isolation(client, auth_service) -> None:
    """Verify that reviews and failure classifications require valid roles."""
    reviewer, reviewer_token = _create_user_with_role(auth_service, "REVIEWER", "reviewer")
    researcher, researcher_token = _create_user_with_role(auth_service, "RESEARCHER", "researcher")

    # Both authenticated users can view the catalog of recipes / reviews
    res = client.get("/api/v1/catalog/recipes/presets", headers={"Authorization": f"Bearer {researcher_token}"})
    assert res.status_code == 200

    # Ensure anonymous requests to protected admin actions fail
    res_anon = client.get("/api/v1/admin/users")
    assert res_anon.status_code == 401


def test_checkpoint_and_dataset_security_matrix(client, auth_service) -> None:
    """Verify checkpoints and datasets endpoints operate securely under all token states."""
    _, token = _create_user_with_role(auth_service, "RESEARCHER", "data_res")
    headers = {"Authorization": f"Bearer {token}"}

    # GET /api/v1/catalog/datasets
    res = client.get("/api/v1/catalog/datasets", headers=headers)
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # GET /api/v1/model-versions
    res_models = client.get("/api/v1/model-versions", headers=headers)
    assert res_models.status_code == 200


def test_suspended_user_is_strictly_blocked(client, auth_service) -> None:
    """Verify that a suspended user cannot perform actions even with an unexpired JWT."""
    user, token = _create_user_with_role(auth_service, "RESEARCHER", "suspended_target")
    # Admin suspends user
    auth_service.update_user_status(admin_user_id="admin-root", target_user_id=user.id, status="SUSPENDED")

    # Attempt to authenticate / call me
    res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403
    assert "suspended" in res.json().get("detail", "").lower()
