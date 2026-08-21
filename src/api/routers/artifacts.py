"""Project-scoped artifact upload, direct-download, and metadata routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from src.api.platform_dependencies import get_platform_artifacts, require_platform_actor
from src.api.schemas.platform import CreateArtifactUploadSessionIn, FinalizeArtifactUploadIn
from src.storage.service import ArtifactService

router = APIRouter(prefix="/projects/{project_id}", tags=["Artifacts"])


@router.post("/artifact-upload-sessions", status_code=status.HTTP_201_CREATED)
async def create_upload_session(
    project_id: str,
    body: CreateArtifactUploadSessionIn,
    actor_id: str = Depends(require_platform_actor),
    artifacts: ArtifactService = Depends(get_platform_artifacts),
) -> dict:
    return artifacts.begin_upload(project_id=project_id, actor_id=actor_id, **body.model_dump())


@router.put("/artifact-upload-sessions/{session_id}/content", status_code=status.HTTP_204_NO_CONTENT)
async def upload_local_content(
    project_id: str,
    session_id: str,
    request: Request,
    actor_id: str = Depends(require_platform_actor),
    artifacts: ArtifactService = Depends(get_platform_artifacts),
) -> Response:
    del actor_id
    try:
        artifacts.upload_local_content(project_id, session_id, await request.body())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/artifact-upload-sessions/{session_id}/complete")
async def complete_upload(
    project_id: str,
    session_id: str,
    body: FinalizeArtifactUploadIn,
    actor_id: str = Depends(require_platform_actor),
    artifacts: ArtifactService = Depends(get_platform_artifacts),
) -> dict:
    del actor_id
    try:
        return _public_artifact(artifacts.complete_upload(project_id, session_id, body.sha256, body.size_bytes))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    project_id: str,
    artifact_id: str,
    actor_id: str = Depends(require_platform_actor),
    artifacts: ArtifactService = Depends(get_platform_artifacts),
) -> dict:
    del actor_id
    artifact = artifacts.get(project_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="ARTIFACT_UNKNOWN")
    return _public_artifact(artifact)


@router.post("/artifacts/{artifact_id}/download-url")
async def create_download_url(
    project_id: str,
    artifact_id: str,
    actor_id: str = Depends(require_platform_actor),
    artifacts: ArtifactService = Depends(get_platform_artifacts),
) -> dict:
    del actor_id
    url = artifacts.signed_download_url(project_id, artifact_id)
    if url is None:
        raise HTTPException(status_code=409, detail="ARTIFACT_NOT_READY")
    if url.startswith("local://"):
        url = f"/api/v1/projects/{project_id}/artifacts/{artifact_id}/content"
    return {"url": url}


@router.get("/artifacts/{artifact_id}/content")
async def download_local_content(
    project_id: str,
    artifact_id: str,
    actor_id: str = Depends(require_platform_actor),
    artifacts: ArtifactService = Depends(get_platform_artifacts),
) -> Response:
    del actor_id
    artifact = artifacts.get(project_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="ARTIFACT_UNKNOWN")
    try:
        content = artifacts.read_bytes(project_id, artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=content, media_type=artifact["mime_type"], headers={"Content-Disposition": f'attachment; filename="{artifact["original_filename"]}"'})


def _public_artifact(artifact: dict) -> dict:
    return {key: value for key, value in artifact.items() if key not in {"storage_key", "created_by_user_id"}}
