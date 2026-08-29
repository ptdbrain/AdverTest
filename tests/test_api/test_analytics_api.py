"""Integration tests for the /analytics/* REST API endpoints."""

from __future__ import annotations

import asyncio

import pytest


async def _create_completed_benchmark_run(client, attack_name: str = "gaussian_noise") -> str:
    """Helper to start and wait for a test run to complete."""
    created = await client.post(
        "/api/v1/runs",
        json={"attacks": [attack_name], "severities": [1], "limit": 2},
    )
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    for _ in range(300):
        item = await client.get(f"/api/v1/runs/{run_id}")
        assert item.status_code == 200
        if item.json()["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            assert item.json()["status"] == "COMPLETED", item.json()
            return run_id
        await asyncio.sleep(0.05)
    raise AssertionError("Run job did not reach COMPLETED state")


@pytest.mark.asyncio
async def test_run_analytics_endpoints(client) -> None:
    """Test all 5 single-run analytics endpoints."""
    run_id = await _create_completed_benchmark_run(client)

    # 1. Summary
    summary_resp = await client.get(f"/api/v1/analytics/runs/{run_id}/summary")
    assert summary_resp.status_code == 200
    summary_data = summary_resp.json()
    assert summary_data["run_id"] == run_id
    assert "ap_clean" in summary_data
    assert "mean_attack_ap" in summary_data
    assert "robust_score" in summary_data

    # 2. Attacks
    attacks_resp = await client.get(f"/api/v1/analytics/runs/{run_id}/attacks")
    assert attacks_resp.status_code == 200
    attacks_data = attacks_resp.json()
    assert isinstance(attacks_data, list)
    assert len(attacks_data) >= 1
    assert attacks_data[0]["attack"] == "gaussian_noise"

    # 3. Severities
    sev_resp = await client.get(f"/api/v1/analytics/runs/{run_id}/severities")
    assert sev_resp.status_code == 200
    sev_data = sev_resp.json()
    assert isinstance(sev_data, list)
    assert len(sev_data) >= 1
    assert "severity" in sev_data[0]

    # 4. Classes
    classes_resp = await client.get(f"/api/v1/analytics/runs/{run_id}/classes")
    assert classes_resp.status_code == 200
    classes_data = classes_resp.json()
    assert isinstance(classes_data, list)

    # 5. Samples
    samples_resp = await client.get(f"/api/v1/analytics/runs/{run_id}/samples?limit=10")
    assert samples_resp.status_code == 200
    samples_data = samples_resp.json()
    assert "total_samples" in samples_data
    assert "items" in samples_data
    assert isinstance(samples_data["items"], list)


@pytest.mark.asyncio
async def test_comparison_analytics_endpoints(client) -> None:
    """Test all 4 comparison analytics endpoints."""
    base_run_id = await _create_completed_benchmark_run(client)
    cand_run_id = await _create_completed_benchmark_run(client)

    comp_created = await client.post(
        "/api/v1/model-comparisons",
        json={"baseline_run_id": base_run_id, "candidate_run_id": cand_run_id},
    )
    assert comp_created.status_code == 201, f"comp_created: {comp_created.text}"
    comp_json = comp_created.json()
    comparison_id = comp_json.get("comparison_id") or comp_json.get("id")
    assert comparison_id is not None, f"comp_json: {comp_json}"

    # 1. Summary
    summary_resp = await client.get(f"/api/v1/analytics/comparisons/{comparison_id}/summary")
    assert summary_resp.status_code == 200, (
        f"summary_resp {summary_resp.status_code}: {summary_resp.text}, comparison_id={comparison_id}"
    )
    summary_data = summary_resp.json()
    assert summary_data["comparison_id"] == comparison_id
    assert summary_data["paired"] is True
    assert "verdict" in summary_data

    # 2. Recovery
    recovery_resp = await client.get(f"/api/v1/analytics/comparisons/{comparison_id}/recovery")
    assert recovery_resp.status_code == 200
    recovery_data = recovery_resp.json()
    assert "overall_recovery" in recovery_data
    assert "per_attack_recovery" in recovery_data

    # 3. Classes
    classes_resp = await client.get(f"/api/v1/analytics/comparisons/{comparison_id}/classes")
    assert classes_resp.status_code == 200
    classes_data = classes_resp.json()
    assert isinstance(classes_data, list)

    # 4. Failures
    failures_resp = await client.get(f"/api/v1/analytics/comparisons/{comparison_id}/failures")
    assert failures_resp.status_code == 200
    failures_data = failures_resp.json()
    assert "total_baseline_failures" in failures_data
    assert "recovered_count" in failures_data
    assert "still_failed_count" in failures_data


@pytest.mark.asyncio
async def test_analytics_404_handling(client) -> None:
    """Verify clean 404 responses for non-existent entities."""
    resp = await client.get("/api/v1/analytics/runs/non-existent-run/summary")
    assert resp.status_code == 404

    resp = await client.get("/api/v1/analytics/comparisons/non-existent-comp/summary")
    assert resp.status_code == 404
