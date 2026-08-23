from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from uuid import uuid4

from src.core.platform_contracts import ArtifactKind, ArtifactState, JobStatus
from src.jobs.service import PlatformJobService
from src.persistence.database import PlatformDatabase
from src.storage.export_service import AttackedDatasetExportService
from src.storage.local import LocalArtifactStorage
from src.storage.service import ArtifactService


def _services(tmp_path: Path) -> tuple[ArtifactService, PlatformJobService]:
    database = PlatformDatabase(f"sqlite:///{tmp_path / 'platform.db'}")
    database.create_schema()
    artifacts = ArtifactService(database, LocalArtifactStorage(str(tmp_path / "objects")), signed_url_ttl_seconds=900)
    return artifacts, PlatformJobService(database)


def test_upload_session_quarantines_and_hashes_artifact(tmp_path: Path) -> None:
    artifacts, _ = _services(tmp_path)
    project_id, actor_id = str(uuid4()), str(uuid4())
    content = b"checkpoint bytes"
    digest = hashlib.sha256(content).hexdigest()

    session = artifacts.begin_upload(
        project_id=project_id, actor_id=actor_id, kind=ArtifactKind.CHECKPOINT,
        original_filename="model.pt", mime_type="application/octet-stream",
        expected_size_bytes=len(content), expected_sha256=digest,
    )
    artifacts.upload_local_content(project_id, session["upload_session_id"], content)
    artifact = artifacts.complete_upload(project_id, session["upload_session_id"], digest, len(content))

    assert artifact["state"] == ArtifactState.QUARANTINED.value
    assert artifact["sha256"] == digest


def test_job_state_is_durable_cancellable_and_retryable(tmp_path: Path) -> None:
    _, jobs = _services(tmp_path)
    project_id, actor_id = str(uuid4()), str(uuid4())
    job = jobs.create(project_id=project_id, owner_user_id=actor_id, job_type="export", request={}, total_units=2)

    assert jobs.start(job["id"])
    assert not jobs.start(job["id"])
    assert jobs.progress(job["id"], stage="EXPORTING", completed=1, total=2, message="halfway")
    assert jobs.cancel(project_id, job["id"])
    assert jobs.fail(job["id"], "JOB_CANCELLED", "cancelled")
    cancelled = jobs.get(project_id, job["id"])
    assert cancelled is not None and cancelled["status"] == JobStatus.CANCELLED.value
    retried = jobs.retry(project_id, job["id"])
    assert retried is not None and retried["status"] == JobStatus.QUEUED.value


def test_attacked_export_contains_required_portable_files(tmp_path: Path) -> None:
    artifacts, _ = _services(tmp_path)
    project_id, actor_id = str(uuid4()), str(uuid4())
    media = artifacts.create_internal(
        project_id=project_id, actor_id=actor_id, kind=ArtifactKind.EVIDENCE,
        original_filename="sample.png", mime_type="image/png", content=b"image", state=ArtifactState.READY,
    )
    label = artifacts.create_internal(
        project_id=project_id, actor_id=actor_id, kind=ArtifactKind.EVIDENCE,
        original_filename="sample.txt", mime_type="text/plain", content=b"label", state=ArtifactState.READY,
    )
    export = AttackedDatasetExportService(artifacts).run(
        project_id=project_id,
        actor_id=actor_id,
        request={
            "source_dataset_version_id": str(uuid4()), "task": "detection2d", "attack_method": "fog",
            "severity": 3, "seed": 195, "implementation_version": "1.0.0",
            "media_artifact_ids": [media["id"]], "label_artifact_ids": [label["id"]],
            "manifest": {"pairs": [{"source": "sample-1", "output": "sample-1"}]},
            "recipe": {"id": "fog-v1"},
        },
    )
    archive = artifacts.read_bytes(project_id, export["artifact_id"])
    with zipfile.ZipFile(io.BytesIO(archive)) as zip_file:
        assert {"manifest.json", "recipe.json", "provenance.json", "hashes.json"} <= set(zip_file.namelist())
