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
