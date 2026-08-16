from __future__ import annotations

import asyncio

import pytest


async def _completed_benchmark(client) -> str:
    response = await client.post(
        "/api/v1/runs",
        json={"attacks": ["gaussian_noise"], "severities": [1], "limit": 1},
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    for _ in range(100):
        job = await client.get(f"/api/v1/runs/{run_id}")
        assert job.status_code == 200
        if job.json()["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            assert job.json()["status"] == "COMPLETED", job.json()
            return run_id
        await asyncio.sleep(0.02)
    raise AssertionError("benchmark did not reach a terminal state")


@pytest.mark.asyncio
async def test_closed_loop_preserves_evidence_and_blocks_unverified_training(client) -> None:
    """CPU acceptance: benchmark evidence flows to an approved backlog and paired export.

    An absent checkpoint must stop before any external training is queued.
    """
    baseline_run_id = await _completed_benchmark(client)
    candidate_run_id = await _completed_benchmark(client)

    comparison = await client.post(
        "/api/v1/model-comparisons",
        json={"baseline_run_id": baseline_run_id, "candidate_run_id": candidate_run_id},
    )
    assert comparison.status_code == 201
    comparison_id = comparison.json()["comparison_id"]

    assert (await client.get(f"/api/v1/model-comparisons/{comparison_id}/metric-deltas")).status_code == 200
    export = await client.get(f"/api/v1/model-comparisons/{comparison_id}/export?format=json")
    assert export.status_code == 200
    assert export.headers["x-content-sha256"]

    created = await client.post("/api/v1/retraining-backlogs", json={"name": f"recovery-{baseline_run_id}"})
    assert created.status_code == 201
    backlog_id = created.json()["id"]
    assert (await client.post(
        f"/api/v1/retraining-backlogs/{backlog_id}/items",
        json={"failure_id": f"{baseline_run_id}:gaussian_noise:severity-1"},
    )).status_code == 201
    approved = await client.post(f"/api/v1/retraining-backlogs/{backlog_id}/approve")
    assert approved.status_code == 200
    assert approved.json()["failure_ids"]

    training = await client.post(
        "/api/v1/training-runs",
        json={
            "trainer_name": "yolo11",
            "model_version": "missing-yolo-checkpoint",
            "dataset_version_id": "dataset-v1",
            "split_manifest_id": "split-v1",
            "defense_profile_id": "defense-v1",
            "seed": 195,
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.001,
            "metadata": {"backlog_id": backlog_id},
        },
    )
    assert training.status_code == 409
    assert training.json()["detail"] == "WAITING_FOR_ARTIFACTS"
