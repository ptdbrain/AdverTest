from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from src.config import get_settings
from src.pipeline import RunConfig


def _completed_report(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "model": "blob_detector",
        "model_version": "blob-1.0.0",
        "dataset": "synthetic_shapes",
        "n_samples": 1,
        "ap_clean": 1.0,
        "cells": [],
        "heatmap": {},
        "worst_cases": [
            {
                "case_id": "failure-fog-001",
                "sample_id": "sample-001",
                "attack": "fog",
                "severity": 4,
                "degradation_hint": 0.4,
                "failed": True,
            }
        ],
        "skipped": [],
        "sample_results": [],
        "metrics": {},
        "provenance": {},
        "seconds": 0.0,
        "simulation_only": True,
    }


@pytest.mark.asyncio
async def test_closed_loop_start_is_durable_and_retrievable(client) -> None:
    import src.api.routes as routes

    run_id = routes._store.create(
        RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1), project_id=client.default_project_id
    )
    routes._store.complete(run_id, _completed_report(run_id))

    created = await client.post("/api/v1/closed-loop/start", json={"run_id": run_id})

    assert created.status_code == 201, created.text
    payload = created.json()
    assert payload["source_run_id"] == run_id
    assert payload["state"] == "FAILURE_IDENTIFIED"
    assert payload["audit"][0]["state"] == "FAILURE_IDENTIFIED"
    assert payload["audit"][0]["artifact_id"] == "failure-fog-001"
    assert payload["audit"][0]["artifact_hash"]
    assert payload["artifacts"]["source_benchmark_run"] == run_id
    assert payload["artifacts"]["failure_cases"] == ["failure-fog-001"]

    durable_job = routes._workflow_store.get_job(payload["loop_id"])
    assert durable_job is not None
    assert durable_job["job_type"] == "closed_loop"
    assert routes._workflow_store.checkpoints(payload["loop_id"])[-1]["payload"] == payload

    import src.main as main_module

    routes = importlib.reload(routes)
    main_module = importlib.reload(main_module)
    async with AsyncClient(
        transport=ASGITransport(app=main_module.app), base_url="http://reopened", headers=dict(client.headers)
    ) as reopened_client:
        fetched = await reopened_client.get(
            f"/api/v1/closed-loop/{payload['loop_id']}", params={"project_id": client.default_project_id}
        )
    assert fetched.status_code == 200
    assert fetched.json() == payload


@pytest.mark.asyncio
async def test_closed_loop_rejects_an_incomplete_source_run(client) -> None:
    import src.api.routes as routes

    run_id = routes._store.create(
        RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1), project_id=client.default_project_id
    )

    response = await client.post("/api/v1/closed-loop/start", json={"run_id": run_id})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "RUN_NOT_COMPLETED"


@pytest.mark.asyncio
async def test_closed_loop_rejects_unknown_or_failure_free_source(client) -> None:
    import src.api.routes as routes

    unknown = await client.post("/api/v1/closed-loop/start", json={"run_id": "missing"})
    assert unknown.status_code == 404
    assert unknown.json()["detail"]["code"] == "RUN_UNKNOWN"

    missing = await client.post("/api/v1/closed-loop/start", json={})
    assert missing.status_code == 422
    assert missing.json()["detail"][0]["loc"] == ["body", "run_id"]

    run_id = routes._store.create(
        RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1), project_id=client.default_project_id
    )
    report = _completed_report(run_id)
    report["worst_cases"] = []
    routes._store.complete(run_id, report)

    failure_free = await client.post("/api/v1/closed-loop/start", json={"run_id": run_id})
    assert failure_free.status_code == 409
    assert failure_free.json()["detail"]["code"] == "NO_FAILURE_CASES"

    zero_run_id = routes._store.create(
        RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1), project_id=client.default_project_id
    )
    zero_report = _completed_report(zero_run_id)
    zero_report["worst_cases"][0]["failed"] = False
    routes._store.complete(zero_run_id, zero_report)

    zero_degradation = await client.post("/api/v1/closed-loop/start", json={"run_id": zero_run_id})
    assert zero_degradation.status_code == 409
    assert zero_degradation.json()["detail"]["code"] == "NO_FAILURE_CASES"


@pytest.mark.asyncio
async def test_closed_loop_get_rejects_another_workflow_type(client) -> None:
    import src.api.routes as routes

    foreign_id = routes._workflow_store.create_job("training", {"seed": 17, "project_id": client.default_project_id})
    response = await client.get(f"/api/v1/closed-loop/{foreign_id}")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "CLOSED_LOOP_UNKNOWN"


@pytest.mark.asyncio
async def test_evidence_status_scans_real_files_without_promoting_them(client) -> None:
    runs_root = Path(get_settings().runs_root)
    run = runs_root / "yolo_b0" / "run-1"
    weights = run / "weights"
    weights.mkdir(parents=True)
    (run / "args.yaml").write_text("model: yolo11s.pt\n", encoding="utf-8")
    (run / "results.csv").write_text("epoch,metrics/mAP50-95(B)\n0,0.5\n", encoding="utf-8")
    (weights / "best.pt").write_bytes(b"checkpoint-fixture")

    robust_run = runs_root / "yolo_r1" / "export"
    robust_run.mkdir(parents=True)
    (robust_run / "robust.pt").write_bytes(b"robust-checkpoint-fixture")
    (robust_run / "training_summary.json").write_text(
        """{
          "checkpoint": {
            "path": "yolo_r1/export/robust.pt",
            "parent_model_version": "yolo11s-clean-b0"
          },
          "registration": {
            "version_id": "yolo11s-robust-r1",
            "model_id": "yolo11s"
          }
        }""",
        encoding="utf-8",
    )

    response = await client.get("/api/v1/status/evidence")

    assert response.status_code == 200
    component = response.json()["components"]["yolo11s-kitti-clean-b0"]
    assert component["runnable"] is True
    assert component["parent_id"] is None
    assert component["parent_lineage"] == []
    assert component["checkpoint_validated"] is False
    assert component["gate_outcome"] is None
    assert component["evidence_tier"] is None

    robust = response.json()["components"]["yolo11s-kitti-robust-r1"]
    assert "yolo11s-robust-r1" not in response.json()["components"]
    assert robust["parent_id"] == "yolo11s-kitti-clean-b0"
    assert robust["parent_lineage"] == ["yolo11s-kitti-clean-b0"]

    repaired = response.json()["components"]["yolo11s-kitti-repaired-r2-fog"]
    assert repaired["parent_id"] == "yolo11s-kitti-robust-r1"
    assert repaired["parent_lineage"] == [
        "yolo11s-kitti-robust-r1",
        "yolo11s-kitti-clean-b0",
    ]
