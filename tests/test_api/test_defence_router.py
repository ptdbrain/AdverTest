"""Integration tests for the /defence and related training/comparison endpoints in defence router."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_defense_profile_crud(client) -> None:
    profile_payload = {
        "profile_id": "profile-test-1",
        "clean_replay_ratio": 0.5,
        "generated_ratio": 0.5,
        "recipe_ids": ["recipe-1"],
    }
    create_resp = await client.post("/api/v1/defense-profiles", json=profile_payload)
    assert create_resp.status_code == 201

    get_resp = await client.get("/api/v1/defense-profiles/profile-test-1")
    assert get_resp.status_code == 200
    assert get_resp.json()["profile_id"] == "profile-test-1"


@pytest.mark.asyncio
async def test_retraining_backlogs_crud(client) -> None:
    create_resp = await client.post("/api/v1/retraining-backlogs", json={"name": "Sprint 1 Backlog"})
    assert create_resp.status_code == 201
    backlog = create_resp.json()
    backlog_id = backlog["id"]

    item_resp = await client.post(
        f"/api/v1/retraining-backlogs/{backlog_id}/items",
        json={"failure_id": "failure-001"},
    )
    assert item_resp.status_code == 201

    approve_resp = await client.post(f"/api/v1/retraining-backlogs/{backlog_id}/approve")
    assert approve_resp.status_code == 200
    assert approve_resp.json()["status"] == "APPROVED"


@pytest.mark.asyncio
async def test_failure_clusters_crud(client) -> None:
    cluster_payload = {
        "name": "High Blur Failures",
        "member_ids": ["fail-1", "fail-2"],
        "defense_profile_id": "profile-test-1",
    }
    create_resp = await client.post("/api/v1/failure-clusters", json=cluster_payload)
    assert create_resp.status_code == 201
    cluster_id = create_resp.json()["cluster_id"]

    get_resp = await client.get(f"/api/v1/failure-clusters/{cluster_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "High Blur Failures"

    list_resp = await client.get("/api/v1/failure-clusters")
    assert list_resp.status_code == 200
    assert any(c["cluster_id"] == cluster_id for c in list_resp.json())
