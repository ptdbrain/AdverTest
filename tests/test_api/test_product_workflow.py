from __future__ import annotations

import pytest
from PIL import Image


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


@pytest.mark.asyncio
async def test_annotated_folder_import_creates_a_persisted_dataset_version(client, tmp_path):
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (8, 8)).save(root / "images" / "frame.png")
    (root / "labels" / "frame.json").write_text("[]", encoding="utf-8")
    (root / "dataset.json").write_text('{"anonymized": true, "split": "test"}', encoding="utf-8")

    response = await client.post("/api/v1/datasets/import", json={
        "root": str(root), "name": "uploaded-detection", "logical_source_id": "upload-1",
        "input_format": "advertest",
    })

    assert response.status_code == 201
    assert response.json()["version_id"].startswith("dataset-")


@pytest.mark.asyncio
async def test_defense_profile_is_persisted_for_retraining(client):
    body = {"profile_id": "defense-fog", "recipe_ids": ["recipe-fog"], "clean_replay_ratio": 0.5, "generated_ratio": 0.5}

    created = await client.post("/api/v1/defense-profiles", json=body)

    assert created.status_code == 201
    fetched = await client.get("/api/v1/defense-profiles/defense-fog")
    assert fetched.status_code == 200


@pytest.mark.asyncio
async def test_unknown_model_comparison_is_not_silently_created(client):
    response = await client.post("/api/v1/model-comparisons", json={"baseline_run_id": "missing-a", "candidate_run_id": "missing-b"})

    assert response.status_code == 404
    assert "unknown run" in response.json()["detail"]
