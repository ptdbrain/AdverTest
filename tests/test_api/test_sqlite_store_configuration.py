"""Legacy SQLite store must fail loudly for unsupported database URLs."""

from __future__ import annotations

import pytest

from src.api.jobs import SqliteRunStore


def test_sqlite_run_store_rejects_postgresql_url() -> None:
    with pytest.raises(ValueError, match="requires a sqlite"):
        SqliteRunStore("postgresql://advertest:password@database:5432/advertest")
