from __future__ import annotations

import asyncio
import io
import zipfile

import pytest
from PIL import Image


def _jpeg() -> bytes:
    image = io.BytesIO()
    Image.new("RGB", (16, 10), color="white").save(image, format="JPEG")
    return image.getvalue()


def _checkpoint_archive() -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("archive/data.pkl", b"metadata-only")
    return payload.getvalue()


@pytest.mark.asyncio
async def test_annotated_detection_batch_finalizes_to_a_benchmark_ready_dataset(client):
    created = await client.post(
        "/api/v1/uploads/batches",
        json={
            "display_name": "road signs",
            "task_id": "detection2d",
            "class_map": {"car": "Car"},
            "anonymized": True,
        },
    )

    assert created.status_code == 201
    batch_id = created.json()["batch_id"]

    uploaded = await client.post(
        "/api/v1/uploads/images",
        content=_jpeg(),
        headers={
            "x-upload-batch-id": batch_id,
            "x-sample-id": "frame-001",
            "x-filename": "frame.jpg",
        },
    )
    assert uploaded.status_code == 201

    saved = await client.put(
        f"/api/v1/uploads/batches/{batch_id}/annotations/frame-001",
        json={
            "task_id": "detection2d",
            "annotations": [{"class_id": "car", "bbox_xyxy": [1, 2, 12, 8]}],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["validation"]["state"] == "VALID"

    finalized = await client.post(f"/api/v1/uploads/batches/{batch_id}/finalize")

    assert finalized.status_code == 201
    assert finalized.json()["benchmark_ready"] is True
    assert finalized.json()["validation"]["issues"] == []


@pytest.mark.asyncio
async def test_checkpoint_upload_rejects_finetune_without_a_parent(client):
    response = await client.post(
        "/api/v1/checkpoints/uploads",
        content=b"not-a-real-checkpoint",
        headers={
            "x-filename": "candidate.pt",
            "x-task-id": "detection2d",
            "x-model-family-id": "yolo11",
            "x-checkpoint-role": "fine_tuned",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "CHECKPOINT_PARENT_REQUIRED"


@pytest.mark.asyncio
async def test_invalid_checkpoint_moves_through_validation_job_to_rejected(client):
    uploaded = await client.post(
        "/api/v1/checkpoints/uploads",
        content=b"not-a-real-checkpoint",
        headers={
            "x-filename": "broken.pt",
            "x-task-id": "detection2d",
            "x-model-family-id": "yolo11",
        },
    )
    assert uploaded.status_code == 201
    checkpoint_id = uploaded.json()["checkpoint_id"]
    assert uploaded.json()["status"] == "PENDING_VALIDATION"
    assert uploaded.json()["validation_job_id"]

    for _ in range(20):
        record = (await client.get(f"/api/v1/checkpoints/{checkpoint_id}")).json()
        if record["status"] in {"READY", "REJECTED"}:
            break
        await asyncio.sleep(0.02)

    assert record["status"] == "REJECTED"
    assert record["validation_reason"] == "CHECKPOINT_FORMAT_INVALID"


@pytest.mark.asyncio
async def test_ready_base_checkpoint_is_listed_for_its_matching_attack_task_and_family(client):
    uploaded = await client.post(
        "/api/v1/checkpoints/uploads",
        content=_checkpoint_archive(),
        headers={
            "x-filename": "base.pt",
            "x-task-id": "detection2d",
            "x-model-family-id": "yolo11",
            "x-display-name": "User YOLO base",
        },
    )
    assert uploaded.status_code == 201
    checkpoint_id = uploaded.json()["checkpoint_id"]

    for _ in range(20):
        record = (await client.get(f"/api/v1/checkpoints/{checkpoint_id}")).json()
        if record["status"] in {"READY", "REJECTED"}:
            break
        await asyncio.sleep(0.02)
    assert record["status"] == "READY"

    checkpoints = (
        await client.get(
            "/api/v1/base-checkpoints",
            params={
                "task_id": "detection2d",
                "model_family_id": "yolo11",
            },
        )
    ).json()
    listed = next(item for item in checkpoints if item["id"] == checkpoint_id)
    assert listed["model_name"] == "User YOLO base"
    assert listed["checkpoint_validated"] is True


@pytest.mark.asyncio
async def test_segmentation_batch_accepts_image_then_finalizes_valid_polygon(client):
    batch = await client.post(
        "/api/v1/uploads/batches",
        json={
            "display_name": "masks",
            "task_id": "segmentation",
            "class_map": {"car": "Car"},
            "anonymized": True,
        },
    )
    batch_id = batch.json()["batch_id"]
    image = await client.post(
        "/api/v1/uploads/images",
        content=_jpeg(),
        headers={
            "x-upload-batch-id": batch_id,
            "x-sample-id": "frame-001",
            "x-filename": "frame.jpg",
            "x-task-id": "segmentation",
        },
    )
    assert image.status_code == 201

    saved = await client.put(
        f"/api/v1/uploads/batches/{batch_id}/annotations/frame-001",
        json={
            "task_id": "segmentation",
            "annotations": [{"class_id": "car", "polygon": [[1, 1], [10, 1], [6, 8]]}],
        },
    )
    assert saved.status_code == 200
    finalized = await client.post(f"/api/v1/uploads/batches/{batch_id}/finalize")
    assert finalized.status_code == 201


@pytest.mark.asyncio
async def test_defence_run_requires_a_completed_attack_baseline(client):
    response = await client.post(
        "/api/v1/defence-runs",
        json={
            "baseline_run_id": "unknown-run",
            "checkpoint_id": "yolo11s-kitti-clean-b0",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "BASELINE_RUN_UNKNOWN"


@pytest.mark.asyncio
async def test_finalized_standalone_attacked_batch_has_runnable_dataset_params(client):
    batch = await client.post(
        "/api/v1/uploads/batches",
        json={
            "display_name": "fog-only",
            "task_id": "detection2d",
            "class_map": {"car": "Car"},
            "anonymized": True,
            "dataset_kind": "attacked_standalone",
        },
    )
    batch_id = batch.json()["batch_id"]
    await client.post(
        "/api/v1/uploads/images",
        content=_jpeg(),
        headers={
            "x-upload-batch-id": batch_id,
            "x-sample-id": "fog-001",
            "x-filename": "fog.jpg",
        },
    )
    await client.put(
        f"/api/v1/uploads/batches/{batch_id}/annotations/fog-001",
        json={
            "task_id": "detection2d",
            "annotations": [{"class_id": "car", "bbox_xyxy": [1, 1, 12, 8]}],
        },
    )

    finalized = await client.post(f"/api/v1/uploads/batches/{batch_id}/finalize")

    assert finalized.status_code == 201
    dataset = finalized.json()["dataset_version"]
    assert dataset["dataset"] == "folder_dataset"
    assert dataset["dataset_params"]["input_format"] == "advertest"
    assert dataset["paired_comparison_ready"] is False


@pytest.mark.asyncio
async def test_paired_attacked_batch_rejects_manifest_with_an_unknown_attacked_sample(client):
    created = await client.post(
        "/api/v1/uploads/batches",
        json={
            "display_name": "paired fog",
            "task_id": "detection2d",
            "class_map": {"car": "Car"},
            "anonymized": True,
            "dataset_kind": "attacked_paired",
            "attacked_manifest": {
                "clean_dataset_version_id": "dataset-clean",
                "pairs": [{"clean_sample_id": "clean-001", "attacked_sample_id": "not-uploaded"}],
                "attack_name": "fog",
                "attack_version": "1.0",
                "severity": 3,
                "seed": 42,
                "source_hash": "source",
                "ground_truth_hash": "ground-truth",
            },
        },
    )
    assert created.status_code == 201
    validation = created.json()["validation"]

    assert validation["state"] == "INVALID"
    assert {issue["code"] for issue in validation["issues"]} >= {"PAIRED_SAMPLE_UNMAPPED"}


@pytest.mark.asyncio
async def test_dataset_import_job_persists_progress_and_failure_reason(client):
    queued = await client.post(
        "/api/v1/datasets/import-jobs",
        json={
            "root": "missing-folder",
            "name": "missing",
            "logical_source_id": "missing-source",
        },
    )
    assert queued.status_code == 202
    job_id = queued.json()["job_id"]

    for _ in range(20):
        job = (await client.get(f"/api/v1/datasets/import-jobs/{job_id}")).json()
        if job["state"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            break
        await asyncio.sleep(0.02)

    assert job["state"] == "FAILED"
    assert job["error"] == "DATASET_ROOT_MISSING"
    assert 0.0 < job["progress_ratio"] < 1.0
