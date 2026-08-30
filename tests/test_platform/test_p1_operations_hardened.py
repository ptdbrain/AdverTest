"""P1 Operations & Hardening Test Suite.

Covers:
- P1.3: Object storage, Redis queue & Compose configuration standardization and healthchecks.
- P1.4: ZIP export streaming, checksums, safe path sanitization (zero zip-slip), and cross-project isolation.
- P1.5: Workload gating: POST /runs {} empty payload rejection, cost estimation token verification, and confirmed submission.
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.main as main_module
from src.core.platform_contracts import ArtifactKind, ArtifactState
from src.persistence.database import PlatformDatabase
from src.storage.export_service import AttackedDatasetExportService
from src.storage.local import LocalArtifactStorage
from src.storage.service import ArtifactService


@pytest.fixture
def client() -> TestClient:
    test_client = TestClient(main_module.app)
    registration = test_client.post(
        "/api/v1/auth/register",
        json={
            "email": f"operations-{uuid.uuid4().hex[:10]}@example.com",
            "password": "StrongPassword123!",
            "display_name": "Operations Owner",
        },
    )
    assert registration.status_code == 201
    headers = {"Authorization": f"Bearer {registration.json()['access_token']}"}
    project = test_client.post("/api/v1/projects", headers=headers, json={"name": "Operations project"})
    assert project.status_code == 201, project.text
    # All canonical project-scoped routes require the query parameter; headers
    # must never be a scope bypass.
    test_client.headers.update(headers)
    test_client.params = {"project_id": project.json()["id"]}
    return test_client


def test_compose_configuration_standardization_and_healthchecks() -> None:
    """P1.3: Verify docker-compose.yml defines healthchecks, standardized env vars, and health dependencies."""
    compose_path = Path(__file__).resolve().parents[2] / "docker-compose.yml"
    assert compose_path.is_file(), "docker-compose.yml not found"
    content = compose_path.read_text(encoding="utf-8")

    # Check standardized environment variables
    assert "OBJECT_STORAGE_ENDPOINT" in content
    assert "OBJECT_STORAGE_BUCKET" in content
    assert "OBJECT_STORAGE_ACCESS_KEY" in content
    assert "OBJECT_STORAGE_SECRET_KEY" in content
    assert "QUEUE_BACKEND: redis" in content
    assert "REDIS_URL" in content

    # Check healthchecks and healthy conditions
    assert "healthcheck:" in content
    assert "condition: service_healthy" in content
    assert "advertest-platform-worker" in content


def test_zip_export_integrity_and_safe_paths(tmp_path: Path) -> None:
    """P1.4: Verify genuine ZIP export creation, archive structure, hashes, and zip-slip safety."""
    db = PlatformDatabase(f"sqlite:///{tmp_path / 'export_test.db'}")
    db.create_schema()
    artifacts = ArtifactService(db, LocalArtifactStorage(str(tmp_path / "obj")), signed_url_ttl_seconds=300)

    project_id = f"proj-{uuid.uuid4().hex[:8]}"
    actor_id = f"user-{uuid.uuid4().hex[:8]}"

    # Create real media and label artifacts
    img_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR..."
    lbl_content = b"0 0.5 0.5 0.2 0.3"
    media = artifacts.create_internal(
        project_id=project_id,
        actor_id=actor_id,
        kind=ArtifactKind.EVIDENCE,
        original_filename="sample.png",
        mime_type="image/png",
        content=img_content,
        state=ArtifactState.READY,
    )
    label = artifacts.create_internal(
        project_id=project_id,
        actor_id=actor_id,
        kind=ArtifactKind.EVIDENCE,
        original_filename="sample.txt",
        mime_type="text/plain",
        content=lbl_content,
        state=ArtifactState.READY,
    )

    export_service = AttackedDatasetExportService(artifacts)
    export_result = export_service.run(
        project_id=project_id,
        actor_id=actor_id,
        request={
            "source_dataset_version_id": str(uuid.uuid4()),
            "task": "detection2d",
            "attack_method": "fog",
            "severity": 3,
            "seed": 2026,
            "implementation_version": "1.0.0",
            "media_artifact_ids": [media["id"]],
            "label_artifact_ids": [label["id"]],
            "manifest": {"pairs": [{"source": media["id"], "target": label["id"]}]},
            "recipe": {"id": "weather-fog-v1"},
        },
    )

    zip_bytes = artifacts.read_bytes(project_id, export_result["artifact_id"], actor_id=actor_id)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        file_list = zf.namelist()

        # 1. Verify required files are present
        assert "manifest.json" in file_list
        assert "recipe.json" in file_list
        assert "provenance.json" in file_list
        assert "hashes.json" in file_list

        # 2. Verify all paths are relative and clean (Zero Zip-Slip)
        for name in file_list:
            assert not name.startswith("/"), f"Absolute path found in ZIP: {name}"
            assert not name.startswith("\\"), f"Absolute Windows path found in ZIP: {name}"
            assert ".." not in name, f"Path traversal found in ZIP: {name}"

        # 3. Parse JSON files
        manifest_data = json.loads(zf.read("manifest.json").decode())
        provenance_data = json.loads(zf.read("provenance.json").decode())
        hashes_data = json.loads(zf.read("hashes.json").decode())

        assert "pairs" in manifest_data
        assert provenance_data["attack_method"] == "fog"
        assert provenance_data["severity"] == 3
        assert len(hashes_data) >= 2


def test_cross_project_zip_export_is_blocked(tmp_path: Path) -> None:
    """P1.4: Cross-project access to another project's artifacts during export must be rejected."""
    db = PlatformDatabase(f"sqlite:///{tmp_path / 'sec_export.db'}")
    db.create_schema()
    artifacts = ArtifactService(db, LocalArtifactStorage(str(tmp_path / "obj")), signed_url_ttl_seconds=300)

    proj_a = f"proj-a-{uuid.uuid4().hex[:8]}"
    user_a = f"user-a-{uuid.uuid4().hex[:8]}"
    user_b = f"user-b-{uuid.uuid4().hex[:8]}"

    media = artifacts.create_internal(
        project_id=proj_a,
        actor_id=user_a,
        kind=ArtifactKind.EVIDENCE,
        original_filename="secret.png",
        mime_type="image/png",
        content=b"secret",
        state=ArtifactState.READY,
    )

    export_service = AttackedDatasetExportService(artifacts)
    # User B attempting to export User A's artifact from Project A
    with pytest.raises(PermissionError):
        export_service.run(
            project_id=proj_a,
            actor_id=user_b,
            request={
                "source_dataset_version_id": str(uuid.uuid4()),
                "task": "detection2d",
                "attack_method": "fog",
                "severity": 3,
                "seed": 2026,
                "implementation_version": "1.0.0",
                "media_artifact_ids": [media["id"]],
                "label_artifact_ids": [media["id"]],
                "manifest": {},
                "recipe": {},
            },
        )


def test_empty_post_runs_payload_is_rejected(client: TestClient) -> None:
    """P1.5: POST /runs with empty payload {} must be rejected with 422 Unprocessable Entity."""
    res = client.post("/api/v1/runs", json={})
    assert res.status_code == 422, f"Expected 422, got {res.status_code}: {res.text}"
    assert "CONFIRMATION_REQUIRED" in res.text


def test_estimate_and_confirmed_run_flow(client: TestClient) -> None:
    """P1.5: Complete flow of /runs/estimate followed by confirmed run creation."""
    config_payload = {
        "model": "blob_detector",
        "dataset": "synthetic_shapes",
        "attacks": ["gaussian_noise"],
        "severities": [1],
        "limit": 2,
    }

    # 1. Request Estimate
    est_res = client.post("/api/v1/runs/estimate", json=config_payload)
    assert est_res.status_code == 200
    est_data = est_res.json()
    assert "estimate_token" in est_data
    assert est_data["estimate_token"] is not None
    assert "n_cells" in est_data
    assert "cost_units" in est_data
    assert "artifact_storage_estimate_bytes" in est_data
    assert "gpu_cpu_cost_estimate" in est_data

    token = est_data["estimate_token"]

    # 2. Attempt run creation with modified config and old token -> 409 Mismatch
    mismatched_payload = {
        **config_payload,
        "limit": 100,  # Modified after estimate!
        "estimate_token": token,
    }
    mismatch_res = client.post("/api/v1/runs", json=mismatched_payload)
    assert mismatch_res.status_code == 409, f"Expected 409 on token mismatch, got {mismatch_res.status_code}"

    # 3. Create run with matched token -> 202 Accepted
    valid_payload = {
        **config_payload,
        "estimate_token": token,
    }
    run_res = client.post("/api/v1/runs", json=valid_payload)
    assert run_res.status_code == 202, f"Expected 202, got {run_res.status_code}: {run_res.text}"
    run_data = run_res.json()
    assert "run_id" in run_data
    assert run_data["status"] in ("QUEUED", "RUNNING", "COMPLETED")
