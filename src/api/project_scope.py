"""Request-local project scope shared by compatibility routes and run storage."""

from __future__ import annotations

from contextvars import ContextVar

_active_project_id: ContextVar[str | None] = ContextVar("active_project_id", default=None)


def set_active_project_id(project_id: str) -> None:
    _active_project_id.set(project_id)


def get_active_project_id() -> str | None:
    return _active_project_id.get()
