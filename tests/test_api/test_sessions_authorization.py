"""Authorization regression tests for experiment-session evidence."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

import src.main as main_module


def _register_actor(client: TestClient, label: str) -> tuple[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"{label}-{uuid.uuid4().hex[:10]}@example.com",
            "password": "StrongPassword123!",
            "display_name": label,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return body["user"]["id"], body["access_token"]


def _create_owned_project(client: TestClient, token: str) -> str:
    response = client.post(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": f"Security project {uuid.uuid4().hex[:10]}"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _session_payload(session_id: str) -> dict[str, object]:
    return {
        "id": session_id,
        "name": "Scoped evidence session",
        "created_at": "2026-08-30T00:00:00Z",
        "updated_at": "2026-08-30T00:00:00Z",
        "runs": [],
    }


def test_session_evidence_requires_active_project_membership() -> None:
    """Anonymous and cross-project callers cannot enumerate or mutate sessions."""
    client = TestClient(main_module.app)
    _, owner_token = _register_actor(client, "session-owner")
    _, outsider_token = _register_actor(client, "session-outsider")
    project_id = _create_owned_project(client, owner_token)
    session_id = f"session-{uuid.uuid4().hex[:16]}"

    assert client.get("/api/v1/sessions", params={"project_id": project_id}).status_code == 401
    assert (
        client.post(
            "/api/v1/sessions",
            params={"project_id": project_id},
            json=_session_payload(session_id),
        ).status_code
        == 401
    )

    owner_headers = {"Authorization": f"Bearer {owner_token}"}
    created = client.post(
        "/api/v1/sessions",
        params={"project_id": project_id},
        headers=owner_headers,
        json=_session_payload(session_id),
    )
    assert created.status_code == 200, created.text

    outsider_headers = {"Authorization": f"Bearer {outsider_token}"}
    assert (
        client.get(
            f"/api/v1/sessions/{session_id}",
            params={"project_id": project_id},
            headers=outsider_headers,
        ).status_code
        == 403
    )
    assert (
        client.delete(
            f"/api/v1/sessions/{session_id}",
            params={"project_id": project_id},
            headers=outsider_headers,
        ).status_code
        == 403
    )
