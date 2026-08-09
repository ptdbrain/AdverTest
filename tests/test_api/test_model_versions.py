from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_registered_model_versions_have_explicit_runtime_safety(client):
    response = await client.get("/api/v1/model-versions")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    for version in response.json():
        assert {"id", "task", "runnable", "blocked_reason", "training_metadata"} <= set(version)
