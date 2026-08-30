"""HTTP regression coverage for scoped benchmark exports."""

from __future__ import annotations

import io
import json
import uuid
import zipfile

from fastapi.testclient import TestClient

import src.api.dependencies as dependencies_module
import src.main as main_module
from src.pipeline.runner import RunConfig


def _register(client: TestClient, name: str) -> tuple[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"{name}-{uuid.uuid4().hex[:10]}@example.com",
            "password": "StrongPassword123!",
            "display_name": name,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    return body["user"]["id"], body["access_token"]


def _project(client: TestClient, token: str) -> str:
    response = client.post(
        "/api/v1/projects",
        headers={"Authorization": f"Bearer {token}"},
        json={"name": f"Run project {uuid.uuid4().hex[:10]}"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _completed_report(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "model": "Mô hình kiểm thử",
        "model_version": "1.0",
        "dataset": "Dữ liệu kiểm thử",
        "n_samples": 1,
        "ap_clean": 0.8,
        "cells": [],
        "heatmap": {},
        "worst_cases": [],
        "skipped": [],
        "sample_results": [],
        "metrics": {},
        "provenance": {
            "checkpoint_sha256": None,
            "run_config_hash": None,
            "protocol_hash": None,
            "seed": 1,
            "cuda_verified": False,
        },
        "seconds": 0.1,
        "simulation_only": True,
        "benchmark_metrics_available": False,
    }


def test_ineligible_run_exports_are_diagnostic_only_and_cannot_be_promoted() -> None:
    client = TestClient(main_module.app)
    owner_id, owner_token = _register(client, "run-owner")
    _, outsider_token = _register(client, "run-outsider")
    project_id = _project(client, owner_token)

    store = dependencies_module.get_store()
    run_id = store.create(
        RunConfig(attacks=["gaussian_noise"], severities=[1], limit=1),
        project_id=project_id,
        owner_user_id=owner_id,
    )
    store.complete(run_id, _completed_report(run_id))

    endpoint = f"/api/v1/runs/{run_id}/download-zip"
    assert client.get(endpoint, params={"project_id": project_id}).status_code == 401
    assert (
        client.get(
            endpoint,
            params={"project_id": project_id},
            headers={"Authorization": f"Bearer {outsider_token}"},
        ).status_code
        == 403
    )

    response = client.get(
        endpoint,
        params={"project_id": project_id},
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert {"diagnostic_report.json", "run_config.json"} <= set(archive.namelist())
        assert "summary.csv" not in archive.namelist()
        diagnostic = json.loads(archive.read("diagnostic_report.json"))
        assert diagnostic["evidence"]["status"] == "NOT_ELIGIBLE"

    pdf_response = client.get(
        f"/api/v1/runs/{run_id}/download-pdf",
        params={"project_id": project_id},
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert pdf_response.status_code == 200, pdf_response.text
    assert pdf_response.content.startswith(b"%PDF-1.4")
    assert b"NOT ELIGIBLE - NO BENCHMARK CONCLUSION" in pdf_response.content

    promotion_response = client.post(
        f"/api/v1/runs/{run_id}/promote",
        params={"project_id": project_id},
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert promotion_response.status_code == 409, promotion_response.text
    assert promotion_response.json()["detail"]["evidence"]["status"] == "NOT_ELIGIBLE"
