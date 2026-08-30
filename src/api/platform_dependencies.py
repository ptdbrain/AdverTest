"""Dependency providers for the new project-scoped platform routers."""

from __future__ import annotations

import functools

from fastapi import Header, HTTPException

from src.api.checkpoint_service import PlatformCheckpointService
from src.auth.security import decode_access_token
from src.compute.backends import ExternalGPUWorker, LocalWorker, RenderWorker
from src.config import get_settings
from src.jobs.queue import HttpDispatcherQueue, LocalJobQueue, RedisJobQueue
from src.jobs.service import PlatformJobService
from src.jobs.worker import PlatformWorker
from src.persistence.database import PlatformDatabase
from src.storage.export_service import AttackedDatasetExportService
from src.storage.local import LocalArtifactStorage
from src.storage.s3 import S3CompatibleStorage
from src.storage.service import ArtifactService


def require_platform_actor(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """Extract authenticated actor identity.

    Security Hardening:
    - Eliminates identity spoofing via client-supplied headers.
    - Strictly requires a valid JWT Bearer token with authenticated claims.
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()
        claims = decode_access_token(token)
        if not claims or "sub" not in claims:
            raise HTTPException(status_code=401, detail="INVALID_TOKEN: Bearer token is invalid or expired.")
        return claims["sub"]

    raise HTTPException(status_code=401, detail="AUTHENTICATION_REQUIRED: Valid Bearer token required.")


def require_project_member(
    project_id: str,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> str:
    """Validate that actor has valid Bearer token and is an active member or owner of project_id."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="AUTHENTICATION_REQUIRED: Valid Bearer token required.")

    token = authorization.removeprefix("Bearer ").strip()
    claims = decode_access_token(token)
    if not claims or "sub" not in claims:
        raise HTTPException(status_code=401, detail="INVALID_TOKEN: Bearer token is invalid or expired.")

    actor_id = claims["sub"]
    user_role = str(claims.get("role", "")).upper()
    if user_role == "ADMIN":
        return actor_id

    from src.persistence.models import ProjectMembershipRecord, ProjectRecord

    db = get_platform_database()
    with db.session() as session:
        membership = session.query(ProjectMembershipRecord).filter(
            ProjectMembershipRecord.project_id == project_id,
            ProjectMembershipRecord.user_id == actor_id,
            ProjectMembershipRecord.status == "ACTIVE",
        ).first()
        if membership:
            return actor_id

        project = session.query(ProjectRecord).filter(
            ProjectRecord.id == project_id,
            ProjectRecord.owner_user_id == actor_id,
        ).first()
        if project:
            return actor_id

    raise HTTPException(status_code=403, detail="FORBIDDEN: User is not an active member or owner of this project.")


@functools.lru_cache
def get_platform_database() -> PlatformDatabase:
    settings = get_settings()
    database = PlatformDatabase(settings.resolved_platform_database_url)
    if settings.app_env != "production":
        database.create_schema()
    return database


@functools.lru_cache
def get_platform_storage():
    settings = get_settings()
    if settings.object_storage_backend == "local":
        return LocalArtifactStorage(settings.artifact_root)
    return S3CompatibleStorage(
        bucket=settings.object_storage_bucket,
        endpoint_url=settings.resolved_object_storage_endpoint,
        region=settings.object_storage_region,
        access_key_id=settings.resolved_object_storage_access_key,
        secret_access_key=settings.resolved_object_storage_secret_key,
    )


@functools.lru_cache
def get_platform_artifacts() -> ArtifactService:
    settings = get_settings()
    return ArtifactService(
        get_platform_database(), get_platform_storage(), settings.object_storage_signed_url_ttl_seconds
    )


@functools.lru_cache
def get_platform_jobs() -> PlatformJobService:
    return PlatformJobService(get_platform_database())


@functools.lru_cache
def get_platform_checkpoints() -> PlatformCheckpointService:
    settings = get_settings()
    return PlatformCheckpointService(
        get_platform_database(),
        get_platform_artifacts(),
        get_platform_jobs(),
        sandbox_timeout_seconds=settings.checkpoint_validation_timeout_seconds,
        sandbox_memory_mb=settings.checkpoint_validation_memory_mb,
        sandbox_cpu_seconds=settings.checkpoint_validation_cpu_seconds,
        allow_local_sandbox=settings.app_env != "production",
        sandbox_url=settings.checkpoint_sandbox_url,
        sandbox_token=settings.checkpoint_sandbox_token,
    )


@functools.lru_cache
def get_platform_exports() -> AttackedDatasetExportService:
    return AttackedDatasetExportService(get_platform_artifacts())


@functools.lru_cache
def get_platform_worker() -> PlatformWorker:
    return PlatformWorker(get_platform_jobs(), get_platform_checkpoints(), get_platform_exports())


@functools.lru_cache
def get_platform_queue():
    settings = get_settings()
    if settings.queue_backend == "redis":
        if not settings.redis_url:
            raise RuntimeError("REDIS_URL is required when QUEUE_BACKEND=redis")
        return RedisJobQueue(settings.redis_url)
    if settings.queue_backend == "http_dispatcher":
        if not settings.external_queue_dispatch_url:
            raise RuntimeError("EXTERNAL_QUEUE_DISPATCH_URL is required when QUEUE_BACKEND=http_dispatcher")
        return HttpDispatcherQueue(settings.external_queue_dispatch_url, settings.external_queue_dispatch_token)
    return LocalJobQueue(get_platform_worker().process, settings.worker_max_concurrency)


@functools.lru_cache
def get_platform_compute():
    settings = get_settings()
    queue = get_platform_queue()
    if settings.external_gpu_worker_url or settings.queue_backend == "http_dispatcher":
        return ExternalGPUWorker(queue)
    if settings.queue_backend == "redis":
        return RenderWorker(queue)
    return LocalWorker(queue)
