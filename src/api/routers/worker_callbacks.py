"""Authenticated control-plane callbacks used by the GPU worker.

The worker deliberately has no PostgreSQL credentials: the Render API remains
the single writer for jobs, events, and result JSON.
"""

from __future__ import annotations

import secrets
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field

from src.api.platform_dependencies import get_platform_jobs
from src.config import get_settings
from src.jobs.service import PlatformJobService

router = APIRouter(prefix="/internal/worker", tags=["Internal GPU worker"])


def _require_worker_token(authorization: str | None = Header(default=None)) -> None:
    expected = get_settings().worker_callback_token
    received = authorization.removeprefix("Bearer ") if authorization else ""
    if not expected or not secrets.compare_digest(expected, received):
        raise HTTPException(status_code=401, detail="UNAUTHORIZED_WORKER")


class ProgressBody(BaseModel):
    stage: str = Field(min_length=1, max_length=100)
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    message: str = Field(min_length=1, max_length=4000)


class CompleteBody(BaseModel):
    result: dict[str, Any]
    message: str = Field(default="GPU job completed", min_length=1, max_length=4000)


class FailBody(BaseModel):
    error_code: str = Field(min_length=1, max_length=100)
    error_message: str = Field(min_length=1, max_length=4000)


@router.post("/jobs/{job_id}/claim")
def claim_job(
    job_id: str,
    _: None = Depends(_require_worker_token),
    jobs: PlatformJobService = Depends(get_platform_jobs),
) -> dict[str, Any]:
    job = jobs.request_for_worker(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="JOB_NOT_FOUND")
    if job["type"] != "benchmark_run":
        raise HTTPException(status_code=422, detail="GPU_WORKER_JOB_TYPE_UNSUPPORTED")
    if not jobs.start(job_id, message="Cloud Run GPU worker claimed job"):
        raise HTTPException(status_code=409, detail="JOB_NOT_CLAIMABLE")
    return {"id": job["id"], "request": job["request"], "project_id": job["project_id"]}


@router.post("/jobs/{job_id}/progress", status_code=204)
def report_progress(
    job_id: str,
    body: ProgressBody,
    _: None = Depends(_require_worker_token),
    jobs: PlatformJobService = Depends(get_platform_jobs),
) -> Response:
    if not jobs.progress(job_id, **body.model_dump()):
        raise HTTPException(status_code=409, detail="JOB_NOT_ACTIVE")
    return Response(status_code=204)


@router.post("/jobs/{job_id}/complete", status_code=204)
def complete_job(
    job_id: str,
    body: CompleteBody,
    _: None = Depends(_require_worker_token),
    jobs: PlatformJobService = Depends(get_platform_jobs),
) -> Response:
    if not jobs.complete(job_id, body.result, body.message):
        raise HTTPException(status_code=409, detail="JOB_NOT_ACTIVE")
    return Response(status_code=204)


@router.post("/jobs/{job_id}/fail", status_code=204)
def fail_job(
    job_id: str,
    body: FailBody,
    _: None = Depends(_require_worker_token),
    jobs: PlatformJobService = Depends(get_platform_jobs),
) -> Response:
    if not jobs.fail(job_id, body.error_code, body.error_message):
        raise HTTPException(status_code=409, detail="JOB_NOT_ACTIVE")
    return Response(status_code=204)


@router.get("/jobs/{job_id}/cancelled")
def cancel_requested(
    job_id: str,
    _: None = Depends(_require_worker_token),
    jobs: PlatformJobService = Depends(get_platform_jobs),
) -> dict[str, bool]:
    return {"cancel_requested": jobs.cancel_requested(job_id)}
