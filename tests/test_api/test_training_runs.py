from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_training_estimate_is_available_without_starting_external_training(client) -> None:
    """Catch a route that forces an unavailable trainer just to estimate cost."""
    response = await client.post(
        "/api/v1/training-runs/estimate",
        json={
            "trainer_name": "yolo11",
            "model_version": "yolo11s-clean-b0",
            "dataset_version_id": "dataset-v1",
            "split_manifest_id": "split-v1",
            "defense_profile_id": "defense-v1",
            "seed": 17,
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
        },
    )

    assert response.status_code == 200
    assert response.json()["gpu_hours"] >= 0


@pytest.mark.asyncio
async def test_training_start_refuses_a_missing_checkpoint_without_queuing_fake_work(client) -> None:
    """Catch a route that silently turns an absent external model into a fake run."""
    response = await client.post(
        "/api/v1/training-runs",
        json={
            "trainer_name": "yolo11",
            "model_version": "missing-yolo-checkpoint",
            "dataset_version_id": "dataset-v1",
            "split_manifest_id": "split-v1",
            "defense_profile_id": "defense-v1",
            "seed": 17,
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "WAITING_FOR_ARTIFACTS"
