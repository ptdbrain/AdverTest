from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_backlog_requires_items_before_approval_and_locks_afterwards(client) -> None:
    """Catch approval of an empty backlog or mutation after human approval."""
    created = await client.post("/api/v1/retraining-backlogs", json={"name": "fog failures"})
    assert created.status_code == 201
    backlog_id = created.json()["id"]

    empty_approval = await client.post(f"/api/v1/retraining-backlogs/{backlog_id}/approve")
    assert empty_approval.status_code == 409
    assert empty_approval.json()["detail"]["code"] == "BACKLOG_EMPTY"

    added = await client.post(
        f"/api/v1/retraining-backlogs/{backlog_id}/items", json={"failure_id": "failure-fog-17"}
    )
    assert added.status_code == 201
    approved = await client.post(f"/api/v1/retraining-backlogs/{backlog_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"

    locked = await client.post(
        f"/api/v1/retraining-backlogs/{backlog_id}/items", json={"failure_id": "failure-fog-18"}
    )
    assert locked.status_code == 409
    assert locked.json()["detail"]["code"] == "BACKLOG_NOT_DRAFT"
