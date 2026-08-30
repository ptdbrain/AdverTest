from __future__ import annotations

import asyncio

import pytest


async def _completed_run(client) -> str:
    created = await client.post(
        "/api/v1/runs",
        json={"attacks": ["gaussian_noise"], "severities": [1], "limit": 1},
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
    raise AssertionError("benchmark job did not reach a terminal state")


@pytest.mark.asyncio
async def test_simulation_comparison_is_diagnostic_only_and_cannot_export_a_conclusion(client) -> None:
    """Simulation outputs must never become a benchmark report or conclusion export."""
    baseline_run_id = await _completed_run(client)
    candidate_run_id = await _completed_run(client)
    created = await client.post(
        "/api/v1/model-comparisons",
        json={"baseline_run_id": baseline_run_id, "candidate_run_id": candidate_run_id},
    )
    assert created.status_code == 201
    comparison = created.json()
    comparison_id = comparison["comparison_id"]
    assert comparison["eligibility"]["status"] == "NOT_ELIGIBLE"
    assert "SIMULATION_ONLY" in comparison["eligibility"]["reasons"]

    deltas = await client.get(f"/api/v1/model-comparisons/{comparison_id}/metric-deltas")
    recovery = await client.get(f"/api/v1/model-comparisons/{comparison_id}/recovery-report")
    csv_export = await client.get(f"/api/v1/model-comparisons/{comparison_id}/export?format=csv")

    assert deltas.status_code == 200
    assert deltas.json() == []
    assert recovery.status_code == 200
    assert recovery.json()["reason"] == "NOT_ELIGIBLE"
    assert csv_export.status_code == 409
    assert csv_export.json()["detail"]["code"] == "NOT_ELIGIBLE_FOR_CONCLUSION_EXPORT"
