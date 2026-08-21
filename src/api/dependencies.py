"""Shared API dependencies and store accessors for FastAPI dependency injection."""

from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor

from src.api.checkpoint_service import CheckpointValidationService
from src.api.generated_dataset_service import GeneratedDatasetService
from src.api.jobs import LocalRunWorker, SqliteRunStore
from src.api.training_service import TrainingJobService
from src.api.workflow_store import WorkflowJobStore
from src.config import get_settings
from src.pipeline.runner import TestRunner
from src.services.person_d import PersonDServices


@functools.lru_cache()
def get_runner() -> TestRunner:
    """Singleton TestRunner instance."""
    return TestRunner()


@functools.lru_cache()
def get_store() -> SqliteRunStore:
    """Singleton SqliteRunStore for benchmark runs and records."""
    return SqliteRunStore(get_settings().database_url)


@functools.lru_cache()
def get_worker() -> LocalRunWorker:
    """Singleton LocalRunWorker managing background test execution."""
    return LocalRunWorker(get_store(), max_workers=get_settings().worker_max_concurrency)


@functools.lru_cache()
def get_workflow_store() -> WorkflowJobStore:
    """Singleton WorkflowJobStore for multi-step jobs and backlogs."""
    return WorkflowJobStore(get_settings().database_url)


@functools.lru_cache()
def get_training_jobs() -> TrainingJobService:
    """Singleton TrainingJobService for model fine-tuning and repair workers."""
    return TrainingJobService(
        get_workflow_store(),
        PersonDServices.default().training.registry,
        max_workers=get_settings().worker_max_concurrency,
    )


@functools.lru_cache()
def get_generated_datasets() -> GeneratedDatasetService:
    """Singleton GeneratedDatasetService for defense dataset generation."""
    return GeneratedDatasetService(
        get_workflow_store(),
        get_store(),
        get_settings().artifact_root,
        max_workers=get_settings().worker_max_concurrency,
    )


@functools.lru_cache()
def get_checkpoint_validations() -> CheckpointValidationService:
    """Singleton CheckpointValidationService for uploaded model checkpoints."""
    return CheckpointValidationService(
        get_store(),
        get_workflow_store(),
        max_workers=get_settings().worker_max_concurrency,
    )


@functools.lru_cache()
def get_dataset_import_workers() -> ThreadPoolExecutor:
    """Singleton ThreadPoolExecutor for asynchronous dataset imports."""
    return ThreadPoolExecutor(max_workers=1)
