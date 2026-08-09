"""Lookup boundary for immutable model versions."""

from __future__ import annotations

from collections.abc import Iterable

from src.models.versions import ModelVersion, TaskKind


class ModelVersionUnavailableError(LookupError):
    """A version is known but unsafe or impossible to execute."""


class ModelVersionCatalog:
    def __init__(self, versions: Iterable[ModelVersion] = ()) -> None:
        self._versions = {version.id: version for version in versions}

    def list(self, *, task: TaskKind | None = None) -> list[ModelVersion]:
        versions = self._versions.values()
        if task is not None:
            versions = (version for version in versions if version.task == task)
        return sorted(versions, key=lambda version: version.id)

    def get(self, version_id: str) -> ModelVersion:
        try:
            return self._versions[version_id]
        except KeyError as exc:
            raise ModelVersionUnavailableError(f"MODEL_VERSION_UNKNOWN: {version_id}") from exc

    def require_runnable(self, version_id: str) -> ModelVersion:
        version = self.get(version_id)
        if not version.runnable:
            raise ModelVersionUnavailableError(version.blocked_reason or "MODEL_VERSION_NOT_RUNNABLE")
        return version
