from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_backlog_and_training_resources_are_addressable(client) -> None:
    """The UI must be able to reopen a backlog and inspect/cancel a queued job."""
    created = await client.post("/api/v1/retraining-backlogs", json={"name": "reopen me"})
    assert created.status_code == 201
    backlog_id = created.json()["id"]

    reopened = await client.get(f"/api/v1/retraining-backlogs/{backlog_id}")
    assert reopened.status_code == 200
    assert reopened.json()["status"] == "DRAFT"

    missing_job = await client.get("/api/v1/training-runs/not-a-job")
    assert missing_job.status_code == 404
    assert (await client.post("/api/v1/training-runs/not-a-job/cancel")).status_code == 404
    assert (await client.get("/api/v1/training-runs/not-a-job/checkpoints")).status_code == 404
    assert (await client.get("/api/v1/training-runs/not-a-job/events")).status_code == 404
