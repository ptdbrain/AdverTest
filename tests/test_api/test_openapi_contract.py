"""OpenAPI contract test suite validating Wave 4 machine-readable endpoints and legacy compatibility."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_openapi_schema_is_valid_and_contains_all_routes() -> None:
    """Verify that the OpenAPI JSON schema is generated cleanly and includes core routes."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()

    paths = schema.get("paths", {})

    # Wave 4 Endpoint Groups Verification
    required_paths = [
        # Scenario / Recipe
        "/api/v1/attack-recipes",
        "/api/v1/attack-recipes/validate",
        "/api/v1/catalog/recipes/presets",
        "/api/v1/attack-recipes/randomize",
        "/api/v1/attack-recipes/sweep",
        "/api/v1/attack-recipes/preview",

        # Benchmark
        "/api/v1/benchmark/protocols",
        "/api/v1/benchmark/runs",
        "/api/v1/benchmark-runs/{run_id}",
        "/api/v1/benchmark-runs/{run_id}/metrics",
        "/api/v1/benchmark-runs/{run_id}/failures",
        "/api/v1/benchmark-runs/{run_id}/cancel",

        # Training
        "/api/v1/training-runs",
        "/api/v1/training-runs/estimate",
        "/api/v1/training-runs/{job_id}",
        "/api/v1/training-runs/{job_id}/checkpoints",
        "/api/v1/training-runs/{job_id}/cancel",
        "/api/v1/training-dataset-manifests/{manifest_id}",

        # Models
        "/api/v1/model-versions",
        "/api/v1/model-versions/{version_id}",
        "/api/v1/model-versions/{version_id}/lineage",
        "/api/v1/model-versions/{version_id}/benchmark-history",
        "/api/v1/model-versions/{version_id}/gate-evidence",

        # Comparisons
        "/api/v1/model-comparisons",
        "/api/v1/model-comparisons/{comparison_id}",
        "/api/v1/model-comparisons/{comparison_id}/metric-deltas",
        "/api/v1/model-comparisons/{comparison_id}/recovery-report",
        "/api/v1/model-comparisons/{comparison_id}/failures",
        "/api/v1/model-comparisons/{comparison_id}/export",

        # Failures / Review
        "/api/v1/failure-cases",
        "/api/v1/failure-clusters",
        "/api/v1/reviews",
        "/api/v1/defense-profiles",

        # Generated Datasets
        "/api/v1/generated-datasets",
        "/api/v1/generated-datasets/{job_id}/manifest",
        "/api/v1/generated-datasets/{job_id}/variants",
        "/api/v1/generated-datasets/{job_id}/events",
        "/api/v1/generated-datasets/{job_id}/validate",

        # Closed-Loop
        "/api/v1/closed-loop/start",
        "/api/v1/closed-loop/{loop_id}",
        "/api/v1/closed-loop/{loop_id}/advance",

        # Status / Evidence
        "/api/v1/status/evidence",

        # Legacy Transition Compatibility
        "/api/v1/runs",
        "/api/v1/runs/{run_id}",
        "/api/v1/runs/{run_id}/cancel",
    ]

    for path in required_paths:
        assert path in paths, f"Missing required OpenAPI path: {path}"


def test_recipe_preset_catalog_endpoint() -> None:
    """Verify GET /api/v1/catalog/recipes/presets returns valid presets."""
    response = client.get("/api/v1/catalog/recipes/presets")
    assert response.status_code == 200
    presets = response.json()
    assert isinstance(presets, list)
    assert len(presets) >= 3
    preset_ids = [p["preset_id"] for p in presets]
    assert "weather_robustness" in preset_ids
    assert "sensor_fault_suite" in preset_ids


def test_recipe_randomize_endpoint() -> None:
    """Verify POST /api/v1/attack-recipes/randomize creates a valid recipe."""
    response = client.post("/api/v1/attack-recipes/randomize", json={"n_steps": 2})
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert len(data["steps"]) == 2


def test_recipe_sweep_endpoint() -> None:
    """Verify POST /api/v1/attack-recipes/sweep generates severity sweep steps."""
    response = client.post("/api/v1/attack-recipes/sweep", json={"attack_id": "gaussian_noise", "severity_range": [1, 2, 3]})
    assert response.status_code == 201
    data = response.json()
    assert len(data["steps"]) == 3
    assert data["steps"][0]["severity"] == 1


def test_failure_clusters_crud_endpoints() -> None:
    """Verify failure clusters list and creation."""
    create_res = client.post(
        "/api/v1/failure-clusters",
        json={"name": "Fog Failures", "member_ids": ["fog-001", "fog-002"]},
    )
    assert create_res.status_code == 201
    cluster = create_res.json()
    cluster_id = cluster["cluster_id"]

    get_res = client.get(f"/api/v1/failure-clusters/{cluster_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Fog Failures"

    list_res = client.get("/api/v1/failure-clusters")
    assert list_res.status_code == 200
    assert any(c["id"] == cluster_id for c in list_res.json())


def test_model_version_lineage_endpoint() -> None:
    """Verify GET /api/v1/model-versions/{version_id}/lineage returns lineage tree."""
    versions_res = client.get("/api/v1/model-versions")
    assert versions_res.status_code == 200
    versions = versions_res.json()
    assert len(versions) > 0
    version_id = versions[0]["id"]

    response = client.get(f"/api/v1/model-versions/{version_id}/lineage")
    assert response.status_code == 200
    data = response.json()
    assert data["version_id"] == version_id
    assert "children" in data

