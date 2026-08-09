from __future__ import annotations

import pytest

from src.models import scan_model_artifacts


@pytest.mark.asyncio
async def test_registered_model_versions_have_explicit_runtime_safety(client):
    response = await client.get("/api/v1/model-versions")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    for version in response.json():
        assert {"id", "task", "runnable", "blocked_reason", "training_metadata"} <= set(version)


def test_sam_checkpoint_is_discovered_but_blocked_without_evaluation_manifest(tmp_path):
    checkpoint = tmp_path / "sam2" / "sam2_hiera_large.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")

    versions = scan_model_artifacts(tmp_path)

    sam = next(version for version in versions if version.task == "segmentation")
    assert sam.checkpoint_path == str(checkpoint.resolve())
    assert sam.runnable is False
    assert sam.blocked_reason == "WAITING_FOR_ARTIFACTS"
