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
    for field in ("name", "group", "cost_class", "severity_levels", "owner", "params_schema"):
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
async def test_run_returns_report_and_is_retrievable(client):
    body = {"attacks": ["gaussian_noise"], "severities": [1, 5], "limit": 2}
    created = await client.post("/api/v1/runs", json=body)
    assert created.status_code == 200
    report = created.json()
    assert report["simulation_only"] is True
    assert len(report["cells"]) == 2
    assert set(report["heatmap"]) == {"gaussian_noise"}

    fetched = await client.get(f"/api/v1/runs/{report['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == report["run_id"]


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
