"""Project-bound loaders for Defence evidence records."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from src.api.jobs import SqliteRunStore


def require_scoped_run(store: SqliteRunStore, *, run_id: str, project_id: str) -> dict[str, Any]:
    """Load a run only when it is explicitly owned by the requested project."""
    run = store.get_scoped(run_id, project_id=project_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RUN_NOT_FOUND_IN_PROJECT", "run_id": run_id},
        )
    return run


def require_scoped_record(
    store: SqliteRunStore,
    *,
    record_type: str,
    record_id: str,
    project_id: str,
) -> dict[str, Any]:
    """Load an immutable Defence record only from its owning project."""
    record = store.get_record(record_type, record_id)
    if record is None or record.get("project_id") != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "RECORD_NOT_FOUND_IN_PROJECT", "id": record_id},
        )
    return record
