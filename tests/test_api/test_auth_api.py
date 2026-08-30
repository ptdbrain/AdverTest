"""Integration tests for Auth API endpoints."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_user_registration_and_login_flow(client) -> None:
    unique_email = f"lead_{uuid.uuid4().hex[:8]}@advertest.ai"

    # 1. Register user
    reg_res = client.post(
        "/api/v1/auth/register",
        json={
            "email": unique_email,
            "password": "Password123!",
            "display_name": "Lead Engineer",
        },
    )
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert "access_token" in reg_data
    assert reg_data["user"]["email"] == unique_email
    token = reg_data["access_token"]

    # 2. Introspect /auth/me with Bearer token
    me_res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == unique_email

    # 3. Login with credentials
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": "Password123!"},
    )
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()


def test_duplicate_registration_rejected(client) -> None:
    unique_email = f"dup_{uuid.uuid4().hex[:8]}@advertest.ai"

    client.post(
        "/api/v1/auth/register",
        json={
            "email": unique_email,
            "password": "Password123!",
            "display_name": "Duplicate User",
        },
    )
    # Registering same email again fails with 400
    dup_res = client.post(
        "/api/v1/auth/register",
        json={
            "email": unique_email,
            "password": "Password123!",
            "display_name": "Duplicate User Again",
        },
    )
    assert dup_res.status_code == 400
