"""Checkpoint registration and pretrained import endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.checkpoint_service import PlatformCheckpointService
from src.api.platform_dependencies import (
    get_platform_checkpoints,
    get_platform_compute,
    require_project_member,
)
from src.api.schemas.platform import ImportUltralyticsIn, RegisterCheckpointIn
from src.compute.backends import ComputeBackend

router = APIRouter(prefix="/projects/{project_id}/checkpoints", tags=["Checkpoints"])


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def register_checkpoint(
    project_id: str,
    body: RegisterCheckpointIn,
    actor_id: str = Depends(require_project_member),
    checkpoints: PlatformCheckpointService = Depends(get_platform_checkpoints),
    compute: ComputeBackend = Depends(get_platform_compute),
) -> dict:
    try:
        checkpoint, job = checkpoints.register_uploaded(
            project_id=project_id,
            actor_id=actor_id,
            artifact_id=str(body.artifact_id),
            task_id=body.task_id,
            model_family_id=body.model_family_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    compute.dispatch(job["id"])
    return {"checkpoint": checkpoint, "job": job}


@router.post("/imports/ultralytics", status_code=status.HTTP_202_ACCEPTED)
async def import_ultralytics(
    project_id: str,
    body: ImportUltralyticsIn,
    actor_id: str = Depends(require_project_member),
    checkpoints: PlatformCheckpointService = Depends(get_platform_checkpoints),
    compute: ComputeBackend = Depends(get_platform_compute),
) -> dict:
    try:
        job = checkpoints.request_ultralytics_import(project_id=project_id, actor_id=actor_id, model_id=body.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    compute.dispatch(job["id"])
    return {"job": job}


@router.get("/{checkpoint_id}")
async def get_checkpoint(
    project_id: str,
    checkpoint_id: str,
    actor_id: str = Depends(require_project_member),
    checkpoints: PlatformCheckpointService = Depends(get_platform_checkpoints),
) -> dict:
    del actor_id
    checkpoint = checkpoints.get(project_id, checkpoint_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="CHECKPOINT_UNKNOWN")
    return checkpoint


@router.get("/{checkpoint_id}/validation")
async def checkpoint_validation_history(
    project_id: str,
    checkpoint_id: str,
    actor_id: str = Depends(require_project_member),
    checkpoints: PlatformCheckpointService = Depends(get_platform_checkpoints),
) -> dict:
    del actor_id
    checkpoint = checkpoints.get(project_id, checkpoint_id)
    if checkpoint is None:
        raise HTTPException(status_code=404, detail="CHECKPOINT_UNKNOWN")
    return {"checkpoint_id": checkpoint_id, "events": checkpoints.validation_history(project_id, checkpoint_id)}
