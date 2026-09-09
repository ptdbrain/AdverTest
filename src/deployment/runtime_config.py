"""Validate the portable local and isolated Render deployment contracts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_PORTABLE_ROOTS = {
    "DATA_ROOT": "./data",
    "ARTIFACT_ROOT": "./data/artifacts",
    "CHECKPOINT_ROOT": "./data/checkpoints",
}
_API_SERVICE_NAME = "advertest-portable-api-preview"
_FRONTEND_SERVICE_NAME = "advertest-portable-frontend-preview"
_DATABASE_NAME = "advertest-portable-preview-db"


def validate_portable_dotenv(path: Path) -> list[str]:
    """Return portable-example errors without reading or printing secret values."""

    values = _dotenv_values(path)
    errors: list[str] = []
    for key, expected in _PORTABLE_ROOTS.items():
        value = values.get(key)
        if value != expected:
            errors.append(f"{key} must be {expected!r} in the portable example")
        elif _looks_machine_specific(value):
            errors.append(f"{key} must not contain a machine-specific path")
    return errors


def validate_preview_blueprint(path: Path) -> list[str]:
    """Return structural errors for the independent Render preview blueprint."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    services = {item.get("name"): item for item in payload.get("services", []) if isinstance(item, dict)}
    databases = {item.get("name"): item for item in payload.get("databases", []) if isinstance(item, dict)}
    errors: list[str] = []

    api = services.get(_API_SERVICE_NAME)
    frontend = services.get(_FRONTEND_SERVICE_NAME)
    if api is None:
        errors.append(f"missing service {_API_SERVICE_NAME}")
    if frontend is None:
        errors.append(f"missing service {_FRONTEND_SERVICE_NAME}")
    if _DATABASE_NAME not in databases:
        errors.append(f"missing independent database {_DATABASE_NAME}")
    if api is None or frontend is None:
        return errors

    api_env = _env_entries(api)
    frontend_env = _env_entries(frontend)
    _require_value(api_env, "APP_ENV", "production", errors)
    _require_relative_root_values(api_env, errors)
    _require_database_reference(api_env, errors)
    _require_generated_secret(api_env, "JWT_SECRET", errors)
    _require_prompted_secret(api_env, "ADMIN_DEFAULT_PASSWORD", errors)
    _require_value(api_env, "OBJECT_STORAGE_BACKEND", "s3", errors)
    _require_value(api_env, "OBJECT_STORAGE_ENDPOINT_URL", "https://storage.googleapis.com", errors)
    for key in ("OBJECT_STORAGE_BUCKET", "OBJECT_STORAGE_ACCESS_KEY_ID", "OBJECT_STORAGE_SECRET_ACCESS_KEY", "CORS_ORIGINS"):
        _require_prompted_secret(api_env, key, errors)
    _require_prompted_secret(frontend_env, "NEXT_PUBLIC_API_URL", errors)
    return errors


def _dotenv_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _looks_machine_specific(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith(("/Users/", "/home/"))


def _env_entries(service: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("key")): entry
        for entry in service.get("envVars", [])
        if isinstance(entry, dict) and entry.get("key")
    }


def _require_value(entries: dict[str, dict[str, Any]], key: str, expected: str, errors: list[str]) -> None:
    if entries.get(key, {}).get("value") != expected:
        errors.append(f"{key} must be {expected!r}")


def _require_relative_root_values(entries: dict[str, dict[str, Any]], errors: list[str]) -> None:
    for key, expected in _PORTABLE_ROOTS.items():
        _require_value(entries, key, expected, errors)


def _require_database_reference(entries: dict[str, dict[str, Any]], errors: list[str]) -> None:
    reference = entries.get("DATABASE_URL", {}).get("fromDatabase")
    if not isinstance(reference, dict) or reference.get("name") != _DATABASE_NAME or reference.get("property") != "connectionString":
        errors.append("DATABASE_URL must reference the preview database connection string")


def _require_generated_secret(entries: dict[str, dict[str, Any]], key: str, errors: list[str]) -> None:
    if entries.get(key, {}).get("generateValue") is not True:
        errors.append(f"{key} must use generateValue")


def _require_prompted_secret(entries: dict[str, dict[str, Any]], key: str, errors: list[str]) -> None:
    if entries.get(key, {}).get("sync") is not False:
        errors.append(f"{key} must use sync: false")
