"""Legacy routes must inherit the same project boundary as the new run router."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

import src.api.dependencies as dependencies_module
import src.main as main_module
from src.pipeline.runner import RunConfig


def _register(client: TestClient, label: str) -> tuple[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"{label}-{uuid.uuid4().hex[:8]}@example.com",
            "password": "StrongPassword123!",
            "display_name": label,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["user"]["id"], response.json()["access_token"]


def _project(client: TestClient, token: str) -> str:
    response = client.post(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": f"Scoped project {uuid.uuid4().hex[:8]}"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_legacy_run_route_cannot_read_a_foreign_project_run() -> None:
    client = TestClient(main_module.app)
    owner_id, owner_token = _register(client, "legacy-owner")
    _, outsider_token = _register(client, "legacy-outsider")
    owner_project = _project(client, owner_token)
    outsider_project = _project(client, outsider_token)
    run_id = dependencies_module.get_store().create(
        RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1),
        project_id=owner_project,
        owner_user_id=owner_id,
    )

    response = client.get(
        f"/api/v1/benchmark-runs/{run_id}",
        params={"project_id": outsider_project},
        headers={"Authorization": f"Bearer {outsider_token}"},
    )

    assert response.status_code == 404, response.text
