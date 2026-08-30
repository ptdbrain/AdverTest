from __future__ import annotations

import pytest

from src.pipeline import RunConfig


def _report(run_id: str, *, model_version: str = "yolo-b0") -> dict:
    return {
        "run_id": run_id,
        "model": "blob_detector",
        "model_version": model_version,
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
            }
        ],
        "skipped": [],
        "sample_results": [],
        "metrics": {},
        "provenance": {"benchmark_protocol_id": "locked-protocol-001"},
        "seconds": 0.0,
        "simulation_only": True,
    }


async def _advance(client, loop_id: str, target: str, artifact_id: str):
    response = await client.post(
        f"/api/v1/closed-loop/{loop_id}/advance",
        json={"target": target, "artifact_id": artifact_id},
    )
    assert response.status_code == 200, response.text
    assert response.json()["state"] == target
    return response.json()


@pytest.mark.asyncio
async def test_closed_loop_binds_the_full_persisted_artifact_chain(client) -> None:
    import src.api.routes as routes

    source_run_id = routes._store.create(RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1))
    routes._store.complete(source_run_id, _report(source_run_id))
    started = await client.post("/api/v1/closed-loop/start", json={"run_id": source_run_id})
    loop_id = started.json()["loop_id"]

    routes._store.put_record(
        "failure_cluster",
        "cluster-001",
        {
            "cluster_id": "cluster-001",
            "member_ids": ["failure-fog-001"],
            "selection_allowed": True,
        },
    )
    await _advance(client, loop_id, "CLUSTER_FORMED", "cluster-001")

    backlog = routes._workflow_store.create_backlog("fog recovery")
    routes._workflow_store.add_backlog_item(backlog["id"], "failure-fog-001")
    await _advance(client, loop_id, "BACKLOG_CREATED", backlog["id"])
    routes._workflow_store.approve_backlog(backlog["id"])
    await _advance(client, loop_id, "BACKLOG_APPROVED", backlog["id"])

    routes._store.put_record("defense_profile", "defense-001", {"profile_id": "defense-001"})
    await _advance(client, loop_id, "DEFENSE_PROFILED", "defense-001")

    routes._store.put_record(
        "training_dataset_manifest",
        "manifest-001",
        {
            "manifest_id": "manifest-001",
            "defense_profile_id": "defense-001",
            "leakage_report_hash": "leakage-sha256",
            "locked_test_excluded": True,
        },
    )
    await _advance(client, loop_id, "DATASET_MANIFEST_CREATED", "manifest-001")

    training_id = routes._workflow_store.create_job(
        "training",
        {
            "defense_profile_id": "defense-001",
            "metadata": {"training_dataset_manifest_id": "manifest-001"},
        },
    )
    await _advance(client, loop_id, "TRAINING_STARTED", training_id)
    routes._workflow_store.complete_job(
        training_id,
        {
            "exported_checkpoint": {
                "sha256": "checkpoint-sha256",
                "load_valid": True,
            },
            "model_version": {
                "id": "yolo-r1",
                "parent_id": "yolo-b0",
                "checkpoint_hash": "checkpoint-sha256",
            },
        },
    )
    await _advance(client, loop_id, "TRAINING_COMPLETED", training_id)
    await _advance(client, loop_id, "CHECKPOINT_VALIDATED", "checkpoint-sha256")

    routes._store.put_record(
        "checkpoint_gate",
        "gate-001",
        {
            "gate_id": "gate-001",
            "passed": True,
            "paired": True,
            "checkpoint_hash": "checkpoint-sha256",
        },
    )
    await _advance(client, loop_id, "GATE_EVALUATED", "gate-001")
    await _advance(client, loop_id, "MODEL_REGISTERED", "yolo-r1")

    candidate_run_id = routes._store.create(RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1))
    await _advance(client, loop_id, "RE_BENCHMARK_STARTED", candidate_run_id)
    routes._store.complete(candidate_run_id, _report(candidate_run_id, model_version="yolo-r1"))
    await _advance(client, loop_id, "RE_BENCHMARK_COMPLETED", candidate_run_id)

    routes._store.put_record(
        "model_comparison",
        "comparison-001",
        {
            "comparison_id": "comparison-001",
            "baseline_run_id": source_run_id,
            "candidate_run_id": candidate_run_id,
            "paired": True,
            "recovery_report": {"recovery_rate": {"unit": "percent"}},
        },
    )
    final = await _advance(client, loop_id, "RECOVERY_REPORTED", "comparison-001")

    assert final["artifacts"]["recovery_report"] == "comparison-001"
    assert len(final["events"]) == 15  # QUEUED + identified + 13 transitions
    assert routes._workflow_store.get_job(loop_id)["status"] == "RECOVERY_REPORTED"
    assert routes._workflow_store.checkpoints(loop_id)[-1]["payload"] == final


@pytest.mark.asyncio
async def test_closed_loop_invalid_skip_is_rejected_without_mutation(client) -> None:
    import src.api.routes as routes

    run_id = routes._store.create(RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1))
    routes._store.complete(run_id, _report(run_id))
    started = await client.post("/api/v1/closed-loop/start", json={"run_id": run_id})
    loop_id = started.json()["loop_id"]
    events_before = routes._workflow_store.events(loop_id)
    checkpoints_before = routes._workflow_store.checkpoints(loop_id)

    response = await client.post(
        f"/api/v1/closed-loop/{loop_id}/advance",
        json={"target": "TRAINING_STARTED", "artifact_id": "missing"},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "INVALID_CLOSED_LOOP_TRANSITION"
    assert routes._workflow_store.events(loop_id) == events_before
    assert routes._workflow_store.checkpoints(loop_id) == checkpoints_before
