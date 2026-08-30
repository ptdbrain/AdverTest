"""Project bootstrap routes establish the membership required by scoped evidence."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from src.main import app


def test_authenticated_user_can_create_and_list_own_project() -> None:
    client = TestClient(app)
    registration = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"project-owner-{uuid.uuid4().hex[:10]}@example.com",
            "password": "StrongPassword123!",
            "display_name": "Project Owner",
        },
    )
    assert registration.status_code == 201
    headers = {"Authorization": f"Bearer {registration.json()['access_token']}"}

    assert client.get("/api/v1/projects").status_code == 401
    created = client.post("/api/v1/projects", headers=headers, json={"name": "Robustness validation"})
    assert created.status_code == 201, created.text
    project = created.json()
    assert project["membership_role"] == "OWNER"

    listed = client.get("/api/v1/projects", headers=headers)
    assert listed.status_code == 200
    assert any(item["id"] == project["id"] for item in listed.json())
