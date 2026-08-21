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

    for _ in range(100):
        item = await client.get(f"/api/v1/runs/{run_id}")
        assert item.status_code == 200
        if item.json()["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            assert item.json()["status"] == "COMPLETED", item.json()
            return run_id
        await asyncio.sleep(0.02)
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

    # 3. Severity
    severity_resp = await client.get(f"/api/v1/analytics/runs/{run_id}/severity")
    assert severity_resp.status_code == 200
    severity_data = severity_resp.json()
    assert isinstance(severity_data, list)
    assert len(severity_data) >= 1

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
    assert comp_created.status_code == 201
    comparison_id = comp_created.json()["comparison_id"]

    # 1. Summary
    summary_resp = await client.get(f"/api/v1/analytics/comparisons/{comparison_id}/summary")
    assert summary_resp.status_code == 200
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
    """Ensure non-existent runs and comparisons return proper 404 status codes."""
    run_resp = await client.get("/api/v1/analytics/runs/non-existent-run-id/summary")
    assert run_resp.status_code == 404

    comp_resp = await client.get("/api/v1/analytics/comparisons/non-existent-comparison-id/summary")
    assert comp_resp.status_code == 404
