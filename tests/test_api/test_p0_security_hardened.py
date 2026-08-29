"""P0 Security hardening acceptance and regression test suite.

Covers:
- P0.1: Public registration cannot receive or self-assign privileged roles (ADMIN, etc.).
- P0.2: Google SSO token verification, forged payload rejection, and role protection.
- P0.3: Cross-tenant project artifact authorization boundary.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient

from src.auth.security import create_access_token
from src.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_public_registration_ignores_or_rejects_admin_role(client: TestClient) -> None:
    """P0.1: Public registration must never create an ADMIN user, regardless of payload."""
    for evil_role in ["ADMIN", "admin", "Admin", "SYSTEM_ADMIN", "REVIEWER", "OWNER"]:
        email = f"attacker_{uuid.uuid4().hex[:8]}@example.com"
        res = client.post(
            "/api/v1/auth/register",
            json={
                "email": email,
                "password": "StrongPassword123!",
                "display_name": "Attacker",
                "role": evil_role,
            },
        )
        assert res.status_code in (201, 422), f"Expected 201 or 422, got {res.status_code}: {res.text}"
        if res.status_code == 201:
            data = res.json()
            assigned_role = data.get("user", {}).get("role")
            assert assigned_role in ("RESEARCHER", "USER", "ENGINEER"), f"Role elevation allowed: {assigned_role}"
            assert assigned_role.upper() != "ADMIN"

            token = data["access_token"]
            # Verify user cannot access admin endpoints
            admin_res = client.get("/api/v1/admin/users", headers={"Authorization": f"Bearer {token}"})
            assert admin_res.status_code == 403, f"Expected 403 on admin endpoint, got {admin_res.status_code}"


def test_public_registration_default_role_is_non_privileged(client: TestClient) -> None:
    """P0.1: Normal public registration without role assigns default non-privileged role."""
    email = f"user_{uuid.uuid4().hex[:8]}@example.com"
    res = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": "StrongPassword123!",
            "display_name": "Regular User",
        },
    )
    assert res.status_code == 201
    data = res.json()
    role = data["user"]["role"]
    assert role in ("RESEARCHER", "USER", "ENGINEER")
    assert role.upper() != "ADMIN"

    token = data["access_token"]
    admin_res = client.get("/api/v1/admin/audit-logs", headers={"Authorization": f"Bearer {token}"})
    assert admin_res.status_code == 403


def test_google_sso_missing_credential_rejected(client: TestClient) -> None:
    """P0.2: Missing credential returns 401/422."""
    res = client.post(
        "/api/v1/auth/google",
        json={
            "email": "victim@example.com",
            "display_name": "Forged User",
            "role": "ADMIN",
        },
    )
    assert res.status_code in (400, 401, 422)


def test_google_sso_fake_token_rejected(client: TestClient) -> None:
    """P0.2: Forged/fake token returns 401."""
    res = client.post(
        "/api/v1/auth/google",
        json={"credential": "forged.fake.token.value"},
    )
    assert res.status_code == 401


def test_google_sso_expired_token_rejected(client: TestClient) -> None:
    """P0.2: Expired token returns 401."""
    expired_token = json.dumps(
        {
            "iss": "accounts.google.com",
            "sub": "sub123",
            "email": "user@example.com",
            "email_verified": True,
            "exp": 1000000000,  # Far past
        }
    )
    res = client.post(
        "/api/v1/auth/google",
        json={"credential": f"test-google-token:{expired_token}"},
    )
    assert res.status_code == 401


def test_google_sso_wrong_issuer_rejected(client: TestClient) -> None:
    """P0.2: Token with wrong issuer returns 401."""
    bad_iss_token = json.dumps(
        {
            "iss": "evil.issuer.com",
            "sub": "sub123",
            "email": "user@example.com",
            "email_verified": True,
            "exp": 2500000000,
        }
    )
    res = client.post(
        "/api/v1/auth/google",
        json={"credential": f"test-google-token:{bad_iss_token}"},
    )
    assert res.status_code == 401


def test_google_sso_unverified_email_rejected(client: TestClient) -> None:
    """P0.2: Token where email_verified is False returns 401."""
    unverified_token = json.dumps(
        {
            "iss": "accounts.google.com",
            "sub": "sub123",
            "email": "unverified@example.com",
            "email_verified": False,
            "exp": 2500000000,
        }
    )
    res = client.post(
        "/api/v1/auth/google",
        json={"credential": f"test-google-token:{unverified_token}"},
    )
    assert res.status_code == 401


def test_google_sso_uses_verified_claims_and_ignores_payload_email_and_role(client: TestClient) -> None:
    """P0.2: System relies exclusively on verified claims, ignoring client payload email and role."""
    verified_email = f"real_verified_{uuid.uuid4().hex[:6]}@example.com"
    valid_token = json.dumps(
        {
            "iss": "accounts.google.com",
            "sub": "sub_valid_999",
            "email": verified_email,
            "email_verified": True,
            "name": "Real Google User",
            "exp": 2500000000,
        }
    )
    res = client.post(
        "/api/v1/auth/google",
        json={
            "credential": f"test-google-token:{valid_token}",
            "email": "fake_attacker@example.com",  # Should be ignored
            "role": "ADMIN",  # Should be ignored
        },
    )
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["user"]["email"] == verified_email
    assert data["user"]["role"] in ("RESEARCHER", "USER")
    assert data["user"]["role"].upper() != "ADMIN"


def test_cross_project_artifact_access_is_blocked(client: TestClient) -> None:
    """P0.3: Actor from Project B must NOT be able to access, download, or create signed URLs for Project A artifacts."""
    proj_a = f"proj-a-{uuid.uuid4().hex[:8]}"
    user_a = f"usr-alice-{uuid.uuid4().hex[:6]}"
    user_b = f"usr-bob-{uuid.uuid4().hex[:6]}"
    token_a = create_access_token({"sub": user_a, "role": "researcher"})
    token_b = create_access_token({"sub": user_b, "role": "researcher"})

    # 1. Unauthenticated request with spoofed X-User-Id header MUST be rejected (401)
    spoof_res = client.post(
        f"/api/v1/projects/{proj_a}/artifact-upload-sessions",
        headers={"X-User-Id": user_a},
        json={
            "kind": "checkpoint",
            "original_filename": "secret_model.pth",
            "mime_type": "application/octet-stream",
            "expected_size_bytes": 12,
            "expected_sha256": None,
        },
    )
    assert spoof_res.status_code == 401, f"Spoofed X-User-Id must be rejected: {spoof_res.text}"

    # 2. User A creates upload session in Project A using authenticated Bearer token
    upload_res = client.post(
        f"/api/v1/projects/{proj_a}/artifact-upload-sessions",
        headers={"Authorization": f"Bearer {token_a}"},
        json={
            "kind": "checkpoint",
            "original_filename": "secret_model.pth",
            "mime_type": "application/octet-stream",
            "expected_size_bytes": 12,
            "expected_sha256": None,
        },
    )
    assert upload_res.status_code == 201, f"Failed upload session: {upload_res.text}"
    session_data = upload_res.json()
    art_id = session_data["artifact_id"]
    sess_id = session_data["upload_session_id"]

    # Upload content
    up_content = client.put(
        f"/api/v1/projects/{proj_a}/artifact-upload-sessions/{sess_id}/content",
        headers={"Authorization": f"Bearer {token_a}"},
        content=b"secret_bytes",
    )
    assert up_content.status_code == 204

    # User B tries to read metadata of Project A's artifact
    art_meta_b = client.get(
        f"/api/v1/projects/{proj_a}/artifacts/{art_id}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert art_meta_b.status_code in (403, 404), f"User B read Project A artifact metadata: {art_meta_b.status_code}"

    # User B tries to download content of Project A's artifact
    art_dl_b = client.get(
        f"/api/v1/projects/{proj_a}/artifacts/{art_id}/content",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert art_dl_b.status_code in (403, 404), f"User B downloaded Project A artifact content: {art_dl_b.status_code}"

    # User B tries to generate signed download URL for Project A's artifact
    art_url_b = client.post(
        f"/api/v1/projects/{proj_a}/artifacts/{art_id}/download-url",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert art_url_b.status_code in (403, 404), f"User B generated signed URL for Project A: {art_url_b.status_code}"


def test_forged_x_user_id_does_not_override_jwt_bearer_identity(client: TestClient) -> None:
    """P0.3: When a valid JWT Bearer token is provided, a spoofed X-User-Id header cannot override identity."""
    # Register genuine User Alice
    email_alice = f"alice_{uuid.uuid4().hex[:6]}@example.com"
    res_alice = client.post(
        "/api/v1/auth/register",
        json={"email": email_alice, "password": "AlicePassword123!", "display_name": "Alice"},
    )
    assert res_alice.status_code == 201
    token_alice = res_alice.json()["access_token"]

    # Spoofed request sending Alice's Bearer token but claiming X-User-Id: admin
    proj_test = f"proj-test-{uuid.uuid4().hex[:8]}"
    upload_res = client.post(
        f"/api/v1/projects/{proj_test}/artifact-upload-sessions",
        headers={"Authorization": f"Bearer {token_alice}", "X-User-Id": "admin_spoofed_id"},
        json={
            "kind": "checkpoint",
            "original_filename": "alice_data.bin",
            "mime_type": "application/octet-stream",
            "expected_size_bytes": 10,
        },
    )
    assert upload_res.status_code == 201
    artifact_id = upload_res.json()["artifact_id"]

    # Verify that the created artifact is owned by Alice, not the spoofed admin
    # Alice can read it
    read_alice = client.get(
        f"/api/v1/projects/{proj_test}/artifacts/{artifact_id}",
        headers={"Authorization": f"Bearer {token_alice}"},
    )
    assert read_alice.status_code == 200


def test_production_startup_validation_fails_on_insecure_secrets() -> None:
    """P0.4: When APP_ENV=production, startup validation must fail if secrets are default, weak, or insecure."""
    from src.config import Settings

    # Insecure default JWT secret
    insecure_settings = Settings(
        app_env="production",
        jwt_secret="advertest-insecure-development-secret-key-2026",
        admin_default_password="AdminPassword123!",
        platform_database_url="postgresql://user:pass@prod-db:5432/advertest",
    )
    with pytest.raises(ValueError) as exc_info:
        insecure_settings.validate_production_environment()

    err_msg = str(exc_info.value)
    assert "JWT_SECRET" in err_msg
    assert "ADMIN_DEFAULT_PASSWORD" in err_msg
    # Ensure error message does not leak the actual secret values
    assert "advertest-insecure-development-secret-key-2026" not in err_msg


def test_production_startup_validation_succeeds_on_hardened_secrets() -> None:
    """P0.4: Production validation succeeds when all secrets are hardened and valid."""
    from src.config import Settings

    secure_settings = Settings(
        app_env="production",
        jwt_secret="k9Y#mP2$vL8!zQ4@xR7^wN1&bT5*hF0~sA3_pD6",
        admin_default_password="K#8vL9!mQ2$zP4@xR7^wN1&bT5*hF0~",
        platform_database_url="postgresql://user:strongpass@prod-db:5432/advertest",
    )
    # Should not raise
    secure_settings.validate_production_environment()


def test_dev_and_test_environments_pass_validation() -> None:
    """P0.4: Test and development environments allow local defaults."""
    from src.config import Settings

    dev_settings = Settings(app_env="development")
    dev_settings.validate_production_environment()

    test_settings = Settings(app_env="test")
    test_settings.validate_production_environment()

