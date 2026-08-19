"""Asynchronous attacked-dataset export route."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from src.api.platform_dependencies import get_platform_compute, get_platform_jobs, require_platform_actor
from src.api.schemas.platform import ExportAttackedDatasetIn
from src.compute.backends import ComputeBackend
from src.jobs.service import PlatformJobService

router = APIRouter(prefix="/projects/{project_id}/exports", tags=["Exports"])


@router.post("/attacked-dataset", status_code=status.HTTP_202_ACCEPTED)
async def export_attacked_dataset(
    project_id: str,
    body: ExportAttackedDatasetIn,
    actor_id: str = Depends(require_platform_actor),
    jobs: PlatformJobService = Depends(get_platform_jobs),
    compute: ComputeBackend = Depends(get_platform_compute),
) -> dict:
    job = jobs.create(
        project_id=project_id,
        owner_user_id=actor_id,
        job_type="attacked_dataset_export",
        request=body.model_dump(mode="json"),
        total_units=2,
    )
    compute.dispatch(job["id"])
    return {"job": job}
