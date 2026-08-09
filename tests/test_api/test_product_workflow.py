from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_recipe_records_are_durable_and_retrievable(client):
    body = {"id": "recipe-fog", "name": "Fog", "seed": 195, "steps": []}

    created = await client.post("/api/v1/attack-recipes", json=body)

    assert created.status_code == 201
    recipe_id = created.json()["id"]
    fetched = await client.get(f"/api/v1/attack-recipes/{recipe_id}")
    assert fetched.status_code == 200
    assert fetched.json()["seed"] == 195


@pytest.mark.asyncio
async def test_model_lineage_exposes_canonical_b_version_or_404(client):
    response = await client.get("/api/v1/model-versions/yolo11s-clean-b0/lineage")

    assert response.status_code in {200, 404}
    if response.status_code == 200:
        assert response.json()["version"]["id"] == "yolo11s-clean-b0"


@pytest.mark.asyncio
async def test_benchmark_run_alias_exposes_job_metrics_and_failures(client):
    created = await client.post("/api/v1/runs", json={"attacks": ["gaussian_noise"], "severities": [1], "limit": 1})
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    job = await client.get(f"/api/v1/benchmark-runs/{run_id}")

    assert job.status_code == 200
    assert (await client.get(f"/api/v1/benchmark-runs/{run_id}/metrics")).status_code in {200, 409}
    assert (await client.get(f"/api/v1/benchmark-runs/{run_id}/failures")).status_code in {200, 409}
