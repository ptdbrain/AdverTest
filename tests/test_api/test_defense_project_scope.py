from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.api.jobs import SqliteRunStore
from src.pipeline.runner import RunConfig


def _config() -> RunConfig:
    return RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1)


def test_scoped_run_loader_never_returns_a_run_owned_by_another_project(tmp_path) -> None:
    from src.api.defense_scope import require_scoped_run

    store = SqliteRunStore(f"sqlite:///{(tmp_path / 'runs.db').as_posix()}")
    run_id = store.create(_config(), project_id="project-a", owner_user_id="owner-a")

    with pytest.raises(HTTPException) as error:
        require_scoped_run(store, run_id=run_id, project_id="project-b")

    assert error.value.status_code == 404
    assert error.value.detail["code"] == "RUN_NOT_FOUND_IN_PROJECT"


def test_scoped_run_loader_preserves_immutable_project_identity(tmp_path) -> None:
    from src.api.defense_scope import require_scoped_run

    store = SqliteRunStore(f"sqlite:///{(tmp_path / 'identity.db').as_posix()}")
    run_id = store.create(_config(), project_id="project-a", owner_user_id="owner-a")

    loaded = require_scoped_run(store, run_id=run_id, project_id="project-a")

    assert loaded["project_id"] == "project-a"
    assert loaded["owner_user_id"] == "owner-a"


def test_scoped_record_loader_rejects_a_record_without_matching_project_id(tmp_path) -> None:
    from src.api.defense_scope import require_scoped_record

    store = SqliteRunStore(f"sqlite:///{(tmp_path / 'records.db').as_posix()}")
    store.put_record("model_comparison", "comparison-a", {"project_id": "project-a"})

    with pytest.raises(HTTPException) as error:
        require_scoped_record(
            store,
            record_type="model_comparison",
            record_id="comparison-a",
            project_id="project-b",
        )

    assert error.value.status_code == 404
    assert error.value.detail["code"] == "RECORD_NOT_FOUND_IN_PROJECT"


@pytest.mark.asyncio
async def test_comparison_read_requires_an_explicit_project_context(client) -> None:
    client.event_hooks["request"].clear()

    response = await client.get("/api/v1/model-comparisons/unknown-comparison")

    assert response.status_code == 422
    assert response.json()["detail"] == "PROJECT_ID_REQUIRED: supply project_id in the canonical route or query."


@pytest.mark.asyncio
async def test_comparison_read_hides_another_projects_record(client) -> None:
    from src.api.dependencies import get_store

    comparison_id = "comparison-project-a"
    get_store().put_record(
        "model_comparison",
        comparison_id,
        {"comparison_id": comparison_id, "project_id": client.default_project_id},
    )
    registration = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "defense-scope-outsider@example.com",
            "password": "StrongPassword123!",
            "display_name": "Outsider",
        },
    )
    assert registration.status_code == 201
    outsider_headers = {"Authorization": f"Bearer {registration.json()['access_token']}"}
    project = await client.post("/api/v1/projects", headers=outsider_headers, json={"name": "Project B"})
    assert project.status_code == 201

    client.event_hooks["request"].clear()
    response = await client.get(
        f"/api/v1/model-comparisons/{comparison_id}",
        params={"project_id": project.json()["id"]},
        headers=outsider_headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "RECORD_NOT_FOUND_IN_PROJECT"
