"""Tests for system runtime specs and standalone defence training script."""

from __future__ import annotations

import subprocess
import sys

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_system_runtime_specs_endpoint():
    """Verify GET /api/v1/system/runtime-specs returns valid hardware introspection data."""
    response = client.get("/api/v1/system/runtime-specs")
    assert response.status_code == 200
    data = response.json()
    assert "has_cuda" in data
    assert "device_target" in data
    assert "recommended_batch_size" in data
    assert "recommended_precision" in data
    assert data["recommended_batch_size"] in (2, 4, 8, 16, 32)
    assert data["recommended_precision"] in ("FP16", "FP32")


def test_download_zip_endpoint_404_for_unknown_run():
    """Verify GET /api/v1/runs/{run_id}/download-zip returns 404 on missing run."""
    response = client.get("/api/v1/runs/non-existent-run-id-999/download-zip")
    assert response.status_code == 404


def test_standalone_train_defence_script_execution():
    """Verify scripts/train_defence.py executes properly and produces a defended checkpoint."""
    cmd = [
        sys.executable,
        "scripts/train_defence.py",
        "--model",
        "tests/fixtures/dummy_model.pt",
        "--dataset",
        "tests/fixtures/dummy_dataset.yaml",
        "--recipe",
        "fgsm,depth_fog",
        "--strategy",
        "adversarial_training",
        "--epochs",
        "2",
        "--batch-size",
        "4",
        "--lr",
        "0.001",
        "--output-dir",
        "tests/fixtures/defended_out",
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "DEFENCE TRAINING COMPLETED SUCCESSFULLY!" in result.stdout


def test_live_inference_endpoint():
    """Verify POST /api/v1/runs/live-inference executes model inference and returns real detections."""
    response = client.post(
        "/api/v1/runs/live-inference",
        json={
            "sample_id": "000000",
            "attack_type": "depth_fog",
            "severity": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "clean_detections" in data
    assert "attacked_detections" in data
    assert "inference_time_clean_ms" in data
    assert "metrics" in data
    assert data["sample_id"] == "000000"
    assert len(data["clean_detections"]) >= 1
