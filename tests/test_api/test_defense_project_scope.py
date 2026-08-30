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


@pytest.mark.asyncio
async def test_defense_profile_and_backlog_are_invisible_outside_their_project(client) -> None:
    profile = await client.post(
        "/api/v1/defense-profiles",
        json={"profile_id": "profile-project-a", "clean_replay_ratio": 0.5, "generated_ratio": 0.5},
    )
    assert profile.status_code == 201, profile.text
    backlog = await client.post("/api/v1/retraining-backlogs", json={"name": "Project A backlog"})
    assert backlog.status_code == 201, backlog.text

    registration = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "defense-scope-backlog-outsider@example.com",
            "password": "StrongPassword123!",
            "display_name": "Outsider",
        },
    )
    outsider_headers = {"Authorization": f"Bearer {registration.json()['access_token']}"}
    outsider_project = await client.post("/api/v1/projects", headers=outsider_headers, json={"name": "Project B"})
    assert outsider_project.status_code == 201
    project_id = outsider_project.json()["id"]

    client.event_hooks["request"].clear()
    profile_read = await client.get(
        "/api/v1/defense-profiles/profile-project-a", params={"project_id": project_id}, headers=outsider_headers
    )
    backlog_read = await client.get(
        f"/api/v1/retraining-backlogs/{backlog.json()['id']}", params={"project_id": project_id}, headers=outsider_headers
    )
    assert profile_read.status_code == 404
    assert backlog_read.status_code == 404


@pytest.mark.asyncio
async def test_risk_assessment_cannot_read_a_review_from_another_project(client) -> None:
    from src.api.dependencies import get_store

    store = get_store()
    run_id = store.create(_config(), project_id=client.default_project_id, owner_user_id="owner-a")
    review_id = store.create_review(
        run_id=run_id,
        attack="fog",
        severity=4,
        degradation=0.45,
        dataset="KITTI",
        model="YOLO11s",
        flagged_by="manual",
    )
    owner_assessment = await client.post(
        "/api/v1/risk-rubric/assess", params={"review_id": review_id}
    )
    assert owner_assessment.status_code == 200, owner_assessment.text

    registration = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "risk-scope-outsider@example.com",
            "password": "StrongPassword123!",
            "display_name": "Outsider",
        },
    )
    outsider_headers = {"Authorization": f"Bearer {registration.json()['access_token']}"}
    outsider_project = await client.post("/api/v1/projects", headers=outsider_headers, json={"name": "Project B"})
    assert outsider_project.status_code == 201

    client.event_hooks["request"].clear()
    outsider_assessment = await client.post(
        "/api/v1/risk-rubric/assess",
        params={"review_id": review_id, "project_id": outsider_project.json()["id"]},
        headers=outsider_headers,
    )
    assert outsider_assessment.status_code == 404


@pytest.mark.asyncio
async def test_retired_unscoped_routes_fail_before_they_can_mutate_state(client) -> None:
    response = await client.post("/api/v1/_deprecated/retraining-backlogs", json={"name": "must not exist"})
    assert response.status_code == 410
    assert response.json()["detail"]["code"] == "PROJECT_SCOPED_ROUTE_REQUIRED"


def test_analytics_comparison_loader_is_project_scoped(tmp_path) -> None:
    from src.api.routers.analytics import _require_comparison

    store = SqliteRunStore(f"sqlite:///{(tmp_path / 'analytics.db').as_posix()}")
    store.put_record("model_comparison", "comparison-a", {"project_id": "project-a"})

    with pytest.raises(HTTPException) as error:
        _require_comparison(store, "comparison-a", project_id="project-b")

    assert error.value.status_code == 404


def test_scoped_workflow_loader_hides_a_closed_loop_from_another_project(tmp_path) -> None:
    from src.api.defense_scope import require_scoped_workflow_job
    from src.api.workflow_store import WorkflowJobStore

    workflow = WorkflowJobStore(f"sqlite:///{(tmp_path / 'workflow.db').as_posix()}")
    job_id = workflow.create_job("closed_loop", {"project_id": "project-a"})

    with pytest.raises(HTTPException) as error:
        require_scoped_workflow_job(workflow, job_id=job_id, project_id="project-b")

    assert error.value.status_code == 404
    assert error.value.detail["code"] == "WORKFLOW_NOT_FOUND_IN_PROJECT"
