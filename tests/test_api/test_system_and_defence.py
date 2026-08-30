"""Tests for system runtime specs and standalone defence training script."""

from __future__ import annotations

import json
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


def test_download_zip_endpoint_requires_authentication():
    """Artifact routes do not disclose run existence to anonymous callers."""
    response = client.get("/api/v1/runs/non-existent-run-id-999/download-zip")
    assert response.status_code == 401


def test_standalone_train_defence_script_refuses_to_fake_a_defended_checkpoint():
    """A missing registered trainer must not manufacture a .pt file or AP evidence."""
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
    assert result.returncode == 2
    assert "TRAINER_NOT_AVAILABLE" in result.stdout


def test_standalone_train_defence_demo_is_explicitly_non_scientific(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_defence.py",
            "--model",
            "tests/fixtures/dummy_model.pt",
            "--output-dir",
            str(tmp_path),
            "--demo",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload["source_kind"] == "demo"
    assert payload["scientific_evidence"] is False
    assert "demo_display_estimates" in payload
    assert not list(tmp_path.glob("*.pt"))


def test_live_inference_endpoint_requires_project_authentication():
    """Visual inference must not create unscoped public artifacts."""
    response = client.post(
        "/api/v1/runs/live-inference",
        json={
            "sample_id": "000000", "attack_type": "depth_fog", "severity": 3, "run_id": "public-run"
        },
    )
    assert response.status_code == 401
