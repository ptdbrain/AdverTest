"""Project-scoped job status, events, cancellation, and retry routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.platform_dependencies import get_platform_compute, get_platform_jobs, require_platform_actor
from src.compute.backends import ComputeBackend
from src.jobs.service import PlatformJobService

router = APIRouter(prefix="/projects/{project_id}/jobs", tags=["Jobs"])


@router.get("/{job_id}")
async def get_job(
    project_id: str,
    job_id: str,
    actor_id: str = Depends(require_platform_actor),
    jobs: PlatformJobService = Depends(get_platform_jobs),
) -> dict:
    del actor_id
    job = jobs.get(project_id, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="JOB_UNKNOWN")
    return job


@router.get("/{job_id}/events")
async def get_job_events(
    project_id: str,
    job_id: str,
    after_sequence: int = -1,
    actor_id: str = Depends(require_platform_actor),
    jobs: PlatformJobService = Depends(get_platform_jobs),
) -> dict:
    del actor_id
    if jobs.get(project_id, job_id) is None:
        raise HTTPException(status_code=404, detail="JOB_UNKNOWN")
    return {"job_id": job_id, "events": jobs.events(project_id, job_id, after_sequence=after_sequence)}


@router.post("/{job_id}/cancel")
async def cancel_job(
    project_id: str,
    job_id: str,
    actor_id: str = Depends(require_platform_actor),
    jobs: PlatformJobService = Depends(get_platform_jobs),
) -> dict:
    del actor_id
    if not jobs.cancel(project_id, job_id):
        raise HTTPException(status_code=404, detail="JOB_UNKNOWN")
    return {"job_id": job_id, "cancel_requested": True}


@router.post("/{job_id}/retry")
async def retry_job(
    project_id: str,
    job_id: str,
    actor_id: str = Depends(require_platform_actor),
    jobs: PlatformJobService = Depends(get_platform_jobs),
    compute: ComputeBackend = Depends(get_platform_compute),
) -> dict:
    del actor_id
    job = jobs.retry(project_id, job_id)
    if job is None:
        raise HTTPException(status_code=409, detail="JOB_RETRY_NOT_AVAILABLE")
    compute.dispatch(job["id"])
    return job
