import asyncio

import pytest


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["simulation_only"] is True
    assert response.headers["x-simulation-only"] == "true"


@pytest.mark.asyncio
async def test_attack_catalog_exposes_owner_and_params(client):
    response = await client.get("/api/v1/catalog/attacks")
    assert response.status_code == 200
    items = response.json()
    assert items, "the catalog must not be empty"
    entry = items[0]
    for field in (
        "name",
        "group",
        "cost_class",
        "severity_levels",
        "owner",
        "params_schema",
        "required_tasks",
        "required_sensors",
        "affected_sensors",
    ):
        assert field in entry


@pytest.mark.asyncio
async def test_attack_catalog_filters_by_group(client):
    response = await client.get("/api/v1/catalog/attacks", params={"group": "D"})
    assert response.status_code == 200
    assert all(item["group"] == "D" for item in response.json())


@pytest.mark.asyncio
async def test_model_and_dataset_catalogs(client):
    models = await client.get("/api/v1/catalog/models")
    datasets = await client.get("/api/v1/catalog/datasets")
    assert "blob_detector" in {item["name"] for item in models.json()}
    assert "pointpillars" not in {item["name"] for item in models.json()}
    assert "bevfusion" not in {item["name"] for item in models.json()}
    assert "synthetic_shapes" in {item["name"] for item in datasets.json()}


@pytest.mark.asyncio
async def test_estimate_before_run(client):
    body = {"attacks": ["gaussian_noise"], "severities": [1, 3], "limit": 2}
    response = await client.post("/api/v1/runs/estimate", json=body)
    assert response.status_code == 200
    estimate = response.json()
    assert estimate["n_cells"] == 2
    assert estimate["n_forward_passes"] > 0


@pytest.mark.asyncio
async def test_run_is_queued_and_report_is_retrievable(client):
    body = {"attacks": ["gaussian_noise"], "severities": [1, 5], "limit": 2}
    created = await client.post("/api/v1/runs", json=body)
    assert created.status_code == 202
    job = created.json()
    assert job["status"] in {"QUEUED", "PREPARING", "INFERENCING", "GENERATING", "EVALUATING", "COMPLETED"}
    for _ in range(50):
        fetched = await client.get(f"/api/v1/runs/{job['run_id']}")
        if fetched.json()["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            break
        await asyncio.sleep(0.05)
    assert fetched.status_code == 200
    completed = fetched.json()
    assert completed["status"] == "COMPLETED", completed.get("error")
    report = completed["report"]
    assert report["simulation_only"] is True
    assert len(report["cells"]) == 2
    assert set(report["heatmap"]) == {"gaussian_noise"}
    samples = await client.get(f"/api/v1/runs/{job['run_id']}/samples", params={"attack": "gaussian_noise"})
    assert len(samples.json()) == 4


@pytest.mark.asyncio
async def test_unknown_attack_is_404(client):
    response = await client.post("/api/v1/runs", json={"attacks": ["no_such_attack"], "limit": 1})
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_unknown_run_is_404(client):
    response = await client.get("/api/v1/runs/deadbeef")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_unknown_config_field_is_422(client):
    response = await client.post("/api/v1/runs", json={"not_a_field": 1})
    assert response.status_code == 422
