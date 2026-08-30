from __future__ import annotations

import asyncio
import io
import zipfile

import pytest
from PIL import Image

from src.models.families import adapter_request


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
    assert modes["detection3d"]["title"] == "3D Object Detection"
    assert modes["detection3d"]["status"] in ("coming_later", "waiting_for_artifacts", "ready")


@pytest.mark.asyncio
async def test_attack_catalog_separates_task_families_base_and_defence_checkpoints(client):
    families = (await client.get("/api/v1/model-families", params={"task_id": "detection2d"})).json()
    assert {item["id"] for item in families} >= {"yolo11", "rtdetr", "faster_rcnn"}
    assert "sam2" not in {item["id"] for item in families}

    base = (
        await client.get(
            "/api/v1/base-checkpoints",
            params={"task_id": "detection2d", "model_family_id": "yolo11"},
        )
    ).json()
    assert base and all(item["checkpoint_role"] == "base" for item in base)
    assert all(item["model_family_id"] == "yolo11" for item in base)

    defence = (await client.get("/api/v1/defence-checkpoints", params={"task_id": "detection2d"})).json()
    # Defence checkpoints are project-owned artifacts. A new project sees an
    # honest empty state rather than global placeholder candidates.
    assert defence == []
    assert not {item["id"] for item in base} & {item["id"] for item in defence}


@pytest.mark.asyncio
async def test_task_catalogs_filter_datasets_and_attacks_by_declared_contract(client):
    detection = (await client.get("/api/v1/catalog/datasets", params={"task_id": "detection2d"})).json()
    segmentation = (await client.get("/api/v1/catalog/datasets", params={"task_id": "segmentation"})).json()
    assert detection and all(item["task_id"] == "detection2d" for item in detection)
    assert segmentation and all(item["task_id"] == "segmentation" for item in segmentation)
    assert all("annotation_schema" in item and "input_schema" in item for item in detection + segmentation)

    white_box = (
        await client.get("/api/v1/catalog/attacks", params={"task_id": "detection2d", "threat_model": "white_box"})
    ).json()
    assert white_box and all(item["threat_model"] == "white_box" for item in white_box)
    assert all("attack_type" in item and "scenario_kind" in item and "task_ids" in item for item in white_box)


@pytest.mark.asyncio
async def test_attack_admission_rejects_a_defence_checkpoint(client):
    response = await client.post(
        "/api/v1/runs/preflight",
        json={
            "checkpoint_id": "yolo11s-kitti-clean-b0",
            "model_family_id": "yolo11",
            "task_id": "detection2d",
            "dataset": "synthetic_shapes",
            "attacks": ["gaussian_noise"],
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "DEFENCE_CHECKPOINT_NOT_ALLOWED_IN_ATTACK"


@pytest.mark.asyncio
async def test_yolo_catalog_marks_proposal_only_dag_unavailable_and_preflight_blocks_recipe(client):
    attacks = (
        await client.get(
            "/api/v1/catalog/attacks",
            params={
                "task_id": "detection2d",
                "model_family_id": "yolo11",
                "checkpoint_id": "yolo11s-base",
                "dataset": "synthetic_shapes",
            },
        )
    ).json()
    dag = next(item for item in attacks if item["name"] == "dag")

    assert dag["available"] is False
    assert dag["reason"] == "missing_model_capability:dense_proposals"

    response = await client.post(
        "/api/v1/runs/preflight",
        json={
            "checkpoint_id": "yolo11s-base",
            "model_family_id": "yolo11",
            "task_id": "detection2d",
            "dataset": "synthetic_shapes",
            "recipe": {
                "name": "proposal-only",
                "steps": [
                    {
                        "position": 0,
                        "attack_name": "dag",
                        "implementation_version": dag["implementation_version"],
                        "severity": 3,
                        "seed": 195,
                        "expected_cost": 1.0,
                    }
                ],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["fatal_errors"] == ["INCOMPATIBLE_RECIPE: dag: missing_model_capability:dense_proposals"]


@pytest.mark.asyncio
async def test_unrunnable_sam_and_3d_model_families_disable_their_attacks(client):
    sam = await client.get(
        "/api/v1/catalog/attacks",
        params={
            "task_id": "segmentation",
            "model_family_id": "sam2",
            "dataset": "bdd100k",
        },
    )
    assert sam.status_code == 200
    assert sam.json()
    assert {item["available"] for item in sam.json()} == {False}
    assert {item["reason"] for item in sam.json()} == {"MODEL_FAMILY_NOT_RUNNABLE:WAITING_FOR_ARTIFACTS"}

    three_d = await client.get(
        "/api/v1/catalog/attacks",
        params={
            "task_id": "detection3d",
            "model_family_id": "centerpoint3d",
            "dataset": "nuscenes",
        },
    )
    assert three_d.status_code == 200
    assert three_d.json() == []  # No registered 3D attack can be selected.


@pytest.mark.asyncio
async def test_pointpillars_upload_preserves_approved_config_but_remains_non_runnable(client):
    from src.api import routes as routes_module
    from src.config import Settings

    checkpoint = io.BytesIO()
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("checkpoint/data.pkl", b"quarantine-test")

    uploaded = await client.post(
        "/api/v1/checkpoints/uploads",
        content=checkpoint.getvalue(),
        headers={
            "x-filename": "pointpillars.pth",
            "x-task-id": "detection3d",
            "x-model-family-id": "pointpillars3d",
            "x-model-config-id": "pointpillars-kitti-3class",
        },
    )

    assert uploaded.status_code == 201
    payload = uploaded.json()
    assert payload["model_config"] == "configs/mmdet3d/pointpillars_hv_secfpn_6x8_160e_kitti-3d-3class.py"

    record = None
    for _ in range(50):
        record = routes_module._store.get_record("checkpoint", payload["checkpoint_id"])
        if record is not None and record["status"] == "READY":
            break
        await asyncio.sleep(0.01)
    assert record is not None
    assert record["status"] == "READY"
    version = next(item for item in routes_module._registered_model_versions() if item.id == payload["checkpoint_id"])

    adapter, params = adapter_request(
        version,
        checkpoint=version.checkpoint_path or "",
        config=type("Run", (), {"confidence_threshold": 0.5})(),
        settings=Settings(),
    )

    assert version.runnable is False
    assert version.blocked_reason == "WAITING_FOR_GPU_VALIDATION"
    assert adapter == "pointpillars"
    assert params["config"] == payload["model_config"]


@pytest.mark.asyncio
async def test_pointpillars_upload_rejects_an_unapproved_config_reference(client):
    response = await client.post(
        "/api/v1/checkpoints/uploads",
        content=b"pointpillars-checkpoint",
        headers={
            "x-filename": "pointpillars.pth",
            "x-task-id": "detection3d",
            "x-model-family-id": "pointpillars3d",
            "x-model-config-id": "C:\\untrusted\\model.py",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "MODEL_FAMILY_CONFIG_INVALID"


@pytest.mark.asyncio
async def test_pointpillars_family_is_product_facing_non_runnable(client):
    from src.api import routes as routes_module

    families = await client.get("/api/v1/model-families", params={"task_id": "detection3d"})
    pointpillars = next(item for item in families.json() if item["id"] == "pointpillars3d")
    assert pointpillars["runnable"] is False
    assert pointpillars["blocked_reason"] == "WAITING_FOR_GPU_VALIDATION"

    availability = routes_module._attack_catalog_availability(
        task_id="detection3d",
        model_family_id="pointpillars3d",
        checkpoint_id=None,
        dataset_name="kitti3d",
    )
    assert availability == ("MODEL_FAMILY_NOT_RUNNABLE:WAITING_FOR_GPU_VALIDATION", {})


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
            "steps": [
                {
                    "position": 0,
                    "attack_name": "gaussian_noise",
                    "implementation_version": noise["implementation_version"],
                    "severity": 3,
                    "seed": 195,
                    "expected_cost": 1.0,
                }
            ],
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
