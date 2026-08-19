"""Register finalized GCS dataset bundles as immutable platform versions."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from src.api.platform_dependencies import get_platform_artifacts, get_platform_database, require_platform_actor
from src.core.platform_contracts import ArtifactKind, ArtifactState
from src.persistence.models import ArtifactRecord, DatasetVersionRecord
from src.storage.service import ArtifactService

router = APIRouter(prefix="/projects/{project_id}/dataset-versions", tags=["Platform datasets"])


class RegisterDatasetBundleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    display_name: str = Field(min_length=1, max_length=200)
    task_id: str = Field(default="detection2d", min_length=1, max_length=100)


def _payload(record: DatasetVersionRecord) -> dict:
    return {
        "id": record.id, "dataset_id": record.dataset_id, "project_id": record.project_id,
        "artifact_id": record.artifact_id, "display_name": record.display_name, "task_id": record.task_id,
        "status": record.status, "schema_hash": record.schema_hash, "sample_count": record.sample_count,
        "manifest": json.loads(record.manifest_json), "created_at": record.created_at, "contract_version": "1.0.0",
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def register_dataset_bundle(
    project_id: str, body: RegisterDatasetBundleIn, actor_id: str = Depends(require_platform_actor),
    database=Depends(get_platform_database), artifacts: ArtifactService = Depends(get_platform_artifacts),
) -> dict:
    with database.session() as session:
        artifact = session.scalar(select(ArtifactRecord).where(ArtifactRecord.id == body.artifact_id, ArtifactRecord.project_id == project_id))
        if artifact is None:
            raise HTTPException(status_code=404, detail="ARTIFACT_UNKNOWN")
        if artifact.kind != ArtifactKind.DATASET_BUNDLE.value or artifact.state != ArtifactState.QUARANTINED.value:
            raise HTTPException(status_code=409, detail="DATASET_ARTIFACT_NOT_QUARANTINED")
        existing = session.scalar(select(DatasetVersionRecord).where(DatasetVersionRecord.project_id == project_id, DatasetVersionRecord.artifact_id == artifact.id))
        if existing:
            return _payload(existing)
    try:
        content = artifacts.read_bytes(project_id, body.artifact_id)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.namelist()
            if not members or any(name.startswith("/") or ".." in name.split("/") for name in members):
                raise ValueError("DATASET_ARCHIVE_INVALID")
            if "manifest.json" not in members:
                raise ValueError("DATASET_MANIFEST_MISSING")
            manifest = json.loads(archive.read("manifest.json"))
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    sample_count = manifest.get("sample_count") if isinstance(manifest, dict) else None
    if not isinstance(sample_count, int) or sample_count < 0:
        sample_count = None
    schema_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record = DatasetVersionRecord(
        id=str(uuid4()), dataset_id=str(uuid4()), project_id=project_id, created_by_user_id=actor_id,
        artifact_id=body.artifact_id, display_name=body.display_name, task_id=body.task_id,
        status="READY", schema_hash=schema_hash, sample_count=sample_count, manifest_json=json.dumps(manifest, sort_keys=True),
    )
    with database.session() as session:
        session.add(record)
    artifacts.set_state(body.artifact_id, ArtifactState.READY)
    return _payload(record)


@router.get("")
def list_dataset_versions(project_id: str, actor_id: str = Depends(require_platform_actor), database=Depends(get_platform_database)) -> list[dict]:
    del actor_id
    with database.session() as session:
        records = session.scalars(select(DatasetVersionRecord).where(DatasetVersionRecord.project_id == project_id).order_by(DatasetVersionRecord.created_at.desc())).all()
    return [_payload(record) for record in records]
