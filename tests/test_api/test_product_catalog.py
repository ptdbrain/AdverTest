from __future__ import annotations

import io

import pytest
from PIL import Image


@pytest.mark.asyncio
async def test_perception_modes_show_sam_handoff_as_waiting(client):
    response = await client.get("/api/v1/perception-modes")

    assert response.status_code == 200
    modes = {item["id"]: item for item in response.json()}
    assert modes["detection2d"]["runnable"] is True
    assert modes["segmentation"]["runnable"] is False
    assert modes["segmentation"]["blocked_reason"] == "WAITING_FOR_ARTIFACTS"
    assert modes["detection2d"]["title"] == "2D Object Detection"
    assert modes["segmentation"]["title"] == "Instance Segmentation"
    assert modes["detection3d"]["status"] == "coming_later"


@pytest.mark.asyncio
async def test_attack_catalog_separates_task_families_base_and_defence_checkpoints(client):
    families = (await client.get("/api/v1/model-families", params={"task_id": "detection2d"})).json()
    assert {item["id"] for item in families} >= {"yolo11", "rtdetr", "faster_rcnn"}
    assert "sam2" not in {item["id"] for item in families}

    base = (await client.get(
        "/api/v1/base-checkpoints",
        params={"task_id": "detection2d", "model_family_id": "yolo11"},
    )).json()
    assert base and all(item["checkpoint_role"] == "base" for item in base)
    assert all(item["model_family_id"] == "yolo11" for item in base)

    defence = (await client.get("/api/v1/defence-checkpoints", params={"task_id": "detection2d"})).json()
    assert defence and {item["checkpoint_role"] for item in defence} >= {"defence_baseline", "fine_tuned", "repaired"}
    assert not {item["id"] for item in base} & {item["id"] for item in defence}


@pytest.mark.asyncio
async def test_task_catalogs_filter_datasets_and_attacks_by_declared_contract(client):
    detection = (await client.get("/api/v1/catalog/datasets", params={"task_id": "detection2d"})).json()
    segmentation = (await client.get("/api/v1/catalog/datasets", params={"task_id": "segmentation"})).json()
    assert detection and all(item["task_id"] == "detection2d" for item in detection)
    assert segmentation and all(item["task_id"] == "segmentation" for item in segmentation)
    assert all("annotation_schema" in item and "input_schema" in item for item in detection + segmentation)

    white_box = (await client.get("/api/v1/catalog/attacks", params={"task_id": "detection2d", "threat_model": "white_box"})).json()
    assert white_box and all(item["threat_model"] == "white_box" for item in white_box)
    assert all("attack_type" in item and "scenario_kind" in item and "task_ids" in item for item in white_box)


@pytest.mark.asyncio
async def test_attack_admission_rejects_a_defence_checkpoint(client):
    response = await client.post("/api/v1/runs/preflight", json={
        "checkpoint_id": "yolo11s-kitti-clean-b0",
        "model_family_id": "yolo11",
        "task_id": "detection2d",
        "dataset": "synthetic_shapes",
        "attacks": ["gaussian_noise"],
    })

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "DEFENCE_CHECKPOINT_NOT_ALLOWED_IN_ATTACK"


@pytest.mark.asyncio
async def test_yolo_catalog_marks_proposal_only_dag_unavailable_and_preflight_blocks_recipe(client):
    attacks = (await client.get("/api/v1/catalog/attacks", params={
        "task_id": "detection2d",
        "model_family_id": "yolo11",
        "checkpoint_id": "yolo11s-base",
        "dataset": "synthetic_shapes",
    })).json()
    dag = next(item for item in attacks if item["name"] == "dag")

    assert dag["available"] is False
    assert dag["reason"] == "missing_model_capability:dense_proposals"

    response = await client.post("/api/v1/runs/preflight", json={
        "checkpoint_id": "yolo11s-base",
        "model_family_id": "yolo11",
        "task_id": "detection2d",
        "dataset": "synthetic_shapes",
        "recipe": {
            "name": "proposal-only",
            "steps": [{
                "position": 0,
                "attack_name": "dag",
                "implementation_version": dag["implementation_version"],
                "severity": 3,
                "seed": 195,
                "expected_cost": 1.0,
            }],
        },
    })

    assert response.status_code == 200
    assert response.json()["fatal_errors"] == ["INCOMPATIBLE_RECIPE: dag: missing_model_capability:dense_proposals"]


@pytest.mark.asyncio
async def test_unrunnable_sam_and_3d_model_families_disable_their_attacks(client):
    sam = await client.get("/api/v1/catalog/attacks", params={
        "task_id": "segmentation",
        "model_family_id": "sam2",
        "dataset": "bdd100k",
    })
    assert sam.status_code == 200
    assert sam.json()
    assert {item["available"] for item in sam.json()} == {False}
    assert {item["reason"] for item in sam.json()} == {"MODEL_FAMILY_NOT_RUNNABLE:WAITING_FOR_ARTIFACTS"}

    three_d = await client.get("/api/v1/catalog/attacks", params={
        "task_id": "detection3d",
        "model_family_id": "centerpoint3d",
        "dataset": "nuscenes",
    })
    assert three_d.status_code == 200
    assert three_d.json() == []  # No registered 3D attack can be selected.


@pytest.mark.asyncio
async def test_localhost_alias_is_allowed_for_frontend_catalog_requests(client):
    response = await client.options(
        "/api/v1/perception-modes",
        headers={
            "Origin": "http://127.0.0.1:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"


@pytest.mark.asyncio
async def test_recipe_validation_returns_explicit_compatibility_errors(client):
    attacks = (await client.get("/api/v1/catalog/attacks")).json()
    noise = next(item for item in attacks if item["name"] == "gaussian_noise")
    body = {
        "recipe": {
            "name": "noise",
            "steps": [{
                "position": 0,
                "attack_name": "gaussian_noise",
                "implementation_version": noise["implementation_version"],
                "severity": 3,
                "seed": 195,
                "expected_cost": 1.0,
            }],
        },
        "task": "detection2d",
        "model_capabilities": [],
        "annotation_types": [],
        "modality": "image",
    }

    response = await client.post("/api/v1/attack-recipes/validate", json=body)

    assert response.status_code == 200
    assert response.json()["valid"] is True


@pytest.mark.asyncio
async def test_upload_endpoint_accepts_a_declared_image_payload(client):
    image = io.BytesIO()
    Image.new("RGB", (8, 8), color="white").save(image, format="JPEG")
    response = await client.post(
        "/api/v1/uploads/images",
        content=image.getvalue(),
        headers={"x-filename": "frame.jpg", "content-type": "image/jpeg"},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "frame.jpg"
    assert "/data/uploads/" in response.json()["path"].replace("\\", "/")


@pytest.mark.asyncio
async def test_raw_image_upload_rejects_a_segmentation_contract_without_masks(client):
    response = await client.post(
        "/api/v1/uploads/images",
        content=b"not-used",
        headers={"x-filename": "frame.jpg", "x-task-id": "segmentation"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "ANNOTATIONS_REQUIRED"


@pytest.mark.asyncio
async def test_checkpoint_upload_records_an_untrusted_artifact_for_validation(client):
    response = await client.post(
        "/api/v1/checkpoints/uploads",
        content=b"not-a-real-checkpoint",
        headers={
            "x-filename": "candidate.pt",
            "x-task-id": "detection2d",
            "x-model-family-id": "yolo11",
            "x-display-name": "candidate",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "PENDING_VALIDATION"
    assert payload["family_id"] == "yolo11"
    assert payload["sha256"]
