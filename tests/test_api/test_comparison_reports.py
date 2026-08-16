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
async def test_paired_comparison_exposes_deltas_recovery_and_real_exports(client) -> None:
    """Catch routes that return the raw comparison instead of report resources."""
    baseline_run_id = await _completed_run(client)
    candidate_run_id = await _completed_run(client)
    created = await client.post(
        "/api/v1/model-comparisons",
        json={"baseline_run_id": baseline_run_id, "candidate_run_id": candidate_run_id},
    )
    assert created.status_code == 201
    comparison_id = created.json()["comparison_id"]

    deltas = await client.get(f"/api/v1/model-comparisons/{comparison_id}/metric-deltas")
    recovery = await client.get(f"/api/v1/model-comparisons/{comparison_id}/recovery-report")
    csv_export = await client.get(f"/api/v1/model-comparisons/{comparison_id}/export?format=csv")
    html_export = await client.get(f"/api/v1/model-comparisons/{comparison_id}/export?format=html")

    assert deltas.status_code == 200
    assert deltas.json()["clean_detection_score"]["unit"] == "ratio"
    assert recovery.status_code == 200
    assert recovery.json()["recovery_rate"]["unit"] == "percent"
    assert csv_export.headers["content-type"].startswith("text/csv")
    assert "clean_detection_score" in csv_export.text
    assert html_export.headers["content-type"].startswith("text/html")
    assert "AdverTest comparison" in html_export.text
