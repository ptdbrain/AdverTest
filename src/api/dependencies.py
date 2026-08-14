import functools
from fastapi import Request
from src.config import get_settings
from src.pipeline.runner import TestRunner
from src.api.jobs import SqliteRunStore, LocalRunWorker

@functools.lru_cache()
def get_runner() -> TestRunner:
    return TestRunner()

@functools.lru_cache()
def get_store() -> SqliteRunStore:
    return SqliteRunStore(get_settings().database_url)

@functools.lru_cache()
def get_worker() -> LocalRunWorker:
    return LocalRunWorker(get_store(), max_workers=get_settings().worker_max_concurrency)

from src.api.workflow_store import WorkflowJobStore
from src.api.generated_dataset_service import GeneratedDatasetService
from concurrent.futures import ThreadPoolExecutor

@functools.lru_cache()
def get_workflow_store() -> WorkflowJobStore:
    return WorkflowJobStore(get_settings().database_url)

@functools.lru_cache()
def get_generated_datasets() -> GeneratedDatasetService:
    return GeneratedDatasetService(
        get_workflow_store(),
        get_store(),
        get_settings().artifact_root,
        max_workers=get_settings().worker_max_concurrency,
    )

@functools.lru_cache()
def get_dataset_import_workers() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1)
