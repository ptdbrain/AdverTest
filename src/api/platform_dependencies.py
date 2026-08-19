"""Dependency providers for the new project-scoped platform routers."""

from __future__ import annotations

import functools

from fastapi import Header, HTTPException

from src.api.checkpoint_service import PlatformCheckpointService
from src.compute.backends import ExternalGPUWorker, LocalWorker, RenderWorker
from src.config import get_settings
from src.jobs.queue import LocalJobQueue, RedisJobQueue
from src.jobs.service import PlatformJobService
from src.jobs.worker import PlatformWorker
from src.persistence.database import PlatformDatabase
from src.storage.export_service import AttackedDatasetExportService
from src.storage.local import LocalArtifactStorage
from src.storage.s3 import S3CompatibleStorage
from src.storage.service import ArtifactService


def require_platform_actor(x_user_id: str = Header(..., alias="X-User-Id")) -> str:
    """Temporary actor boundary until D's authenticated CurrentUser dependency lands."""
    if not x_user_id.strip():
        raise HTTPException(status_code=401, detail="INSUFFICIENT_PERMISSION")
    return x_user_id


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
        endpoint_url=settings.object_storage_endpoint_url,
        region=settings.object_storage_region,
        access_key_id=settings.object_storage_access_key_id,
        secret_access_key=settings.object_storage_secret_access_key,
    )


@functools.lru_cache
def get_platform_artifacts() -> ArtifactService:
    settings = get_settings()
    return ArtifactService(get_platform_database(), get_platform_storage(), settings.object_storage_signed_url_ttl_seconds)


@functools.lru_cache
def get_platform_jobs() -> PlatformJobService:
    return PlatformJobService(get_platform_database())


@functools.lru_cache
def get_platform_checkpoints() -> PlatformCheckpointService:
    settings = get_settings()
    return PlatformCheckpointService(
        get_platform_database(), get_platform_artifacts(), get_platform_jobs(),
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
    return LocalJobQueue(get_platform_worker().process, settings.worker_max_concurrency)


@functools.lru_cache
def get_platform_compute():
    settings = get_settings()
    queue = get_platform_queue()
    if settings.external_gpu_worker_url:
        return ExternalGPUWorker(queue)
    if settings.queue_backend == "redis":
        return RenderWorker(queue)
    return LocalWorker(queue)
