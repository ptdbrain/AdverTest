import asyncio
import io
from pathlib import Path

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
    sample = samples.json()[0]
    assert "clean_image_path" not in sample
    assert set(sample["artifacts"]) == {
        "clean_input_url", "attacked_input_url", "clean_prediction_url", "attacked_prediction_url"
    }
    assert all(value is None or value.startswith("/data/") for value in sample["artifacts"].values())


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


@pytest.mark.asyncio
async def test_upload_rejects_non_image_bytes(client):
    response = await client.post(
        "/api/v1/uploads/images",
        headers={"x-filename": "test.png"},
        content=b"not an image file content",
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_IMAGE"


@pytest.mark.asyncio
async def test_raw_upload_is_not_marked_anonymized(client):
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color="red").save(buf, format="PNG")
    png_bytes = buf.getvalue()

    response = await client.post(
        "/api/v1/uploads/images",
        headers={"x-filename": "sample.png", "x-upload-batch-id": "batch-test-123456"},
        content=png_bytes,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["benchmark_ready"] is False
    assert data["status"] == "RAW"
    assert data["annotation_status"] == "MISSING"
    assert data["batch_id"] == "batch-test-123456"


@pytest.mark.asyncio
async def test_quick_inference_fails_fast_without_a_runnable_checkpoint(client):
    response = await client.post("/api/v1/inference-experiments", json={
        "upload_batch_id": "batch-missing-123",
        "model_version_id": "yolo_b0",
        "attacks": ["gaussian_noise"],
    })
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_quick_inference_accepts_the_runnable_catalog_model_id(client, monkeypatch, tmp_path):
    """The model ID returned by the catalog must be accepted by quick inference."""
    from PIL import Image

    import src.api.routes as routes
    from src.models.versions import ModelVersion
    from src.pipeline import PreflightResult

    actual_checkpoint = tmp_path / "yolo11s-clean-b0.pt"
    actual_checkpoint.write_bytes(b"checkpoint")
    discovered = ModelVersion(
        id="yolo11s-clean-b0",
        model_name="yolo11s",
        task="detection2d",
        checkpoint_path=str(actual_checkpoint),
        checkpoint_hash="test-checkpoint",
        parent_id=None,
        training_metadata={"role": "yolo_b0"},
        runnable=True,
    )
    monkeypatch.setattr(routes, "scan_model_artifacts", lambda _root: [discovered])
    monkeypatch.setattr(
        routes._runner,
        "preflight",
        lambda _config: PreflightResult(compatible=("gaussian_noise",), skipped=()),
    )
    monkeypatch.setattr(routes._worker, "enqueue", lambda _run_id, _config: None)

    catalog = await client.get("/api/v1/model-versions")
    model_id = next(item["id"] for item in catalog.json() if item["runnable"])

    image = io.BytesIO()
    Image.new("RGB", (10, 10), color="red").save(image, format="PNG")
    uploaded = await client.post(
        "/api/v1/uploads/images",
        headers={"x-filename": "sample.png", "x-upload-batch-id": "batch-catalog-model"},
        content=image.getvalue(),
    )
    assert uploaded.status_code == 201

    response = await client.post("/api/v1/inference-experiments", json={
        "upload_batch_id": "batch-catalog-model",
        "model_version_id": model_id,
        "attacks": ["gaussian_noise"],
    })
    assert response.status_code == 202


@pytest.mark.asyncio
async def test_random_recipe_is_seeded_and_replayable(client):
    body = {"n_steps": 3, "seed": 195}
    first = await client.post("/api/v1/attack-recipes/randomize", json=body)
    second = await client.post("/api/v1/attack-recipes/randomize", json=body)
    assert first.status_code == second.status_code == 201
    assert first.json()["seed"] == second.json()["seed"] == 195
    assert first.json()["steps"] == second.json()["steps"]


@pytest.mark.asyncio
async def test_protocol_must_lock_before_benchmark(client):
    proto_res = await client.post(
        "/api/v1/benchmark/protocols",
        json={
            "dataset_version_id": "ds-v1",
            "model_version_id": "mv-v1",
            "recipe": {"steps": [{"attack": "gaussian_noise", "severity": 1}]},
            "task": "detection2d",
            "state": "VALIDATED",
        },
    )
    assert proto_res.status_code == 201
    proto_id = proto_res.json()["id"]

    run_config = {
        "model": "blob_detector",
        "dataset": "synthetic_shapes",
        "attacks": ["gaussian_noise"],
        "severities": [1],
        "limit": 1,
    }
    # Attempting to run non-LOCKED (VALIDATED) protocol fails with 409 PROTOCOL_NOT_LOCKED
    run_res = await client.post(f"/api/v1/benchmark/runs?protocol_id={proto_id}", json=run_config)
    assert run_res.status_code == 409
    assert run_res.json()["detail"]["code"] == "PROTOCOL_NOT_LOCKED"

    # Locking advances VALIDATED protocol to LOCKED
    lock_res = await client.post(f"/api/v1/benchmark/protocols/{proto_id}/lock")
    assert lock_res.status_code == 200
    assert lock_res.json()["state"] == "LOCKED"

    # After lock, run creation succeeds
    run_res2 = await client.post(f"/api/v1/benchmark/runs?protocol_id={proto_id}", json=run_config)
    assert run_res2.status_code == 202
