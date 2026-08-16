"""Wave 0 — dataset import idempotency and conflict detection.

Tests:
- Same payload + identity → returns existing record unchanged
- Same ID + different payload → 409 DATASET_VERSION_CONFLICT
- Isolated per-test databases → no state leakage
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from src.api.jobs import SqliteRunStore

_TEST_ROOT = Path(__file__).resolve().parents[2] / ".pytest-temp-wave0"


@pytest.fixture
def isolated_store() -> SqliteRunStore:
    """Each test gets its own SQLite database in a project-local temp dir."""
    test_dir = _TEST_ROOT / f"store-{uuid.uuid4().hex[:8]}"
    test_dir.mkdir(parents=True, exist_ok=True)
    db = test_dir / "test-store.db"
    store = SqliteRunStore(f"sqlite:///{db}")
    yield store
    shutil.rmtree(test_dir, ignore_errors=True)


class TestDatasetImportIdempotency:
    """Contract: same payload → idempotent; same ID different payload → 409."""

    def test_same_payload_returns_existing(self, isolated_store: SqliteRunStore):
        payload = {"version_id": "v1", "name": "kitti-train", "records": []}
        first = isolated_store.put_record("dataset_version", "v1", payload)
        second = isolated_store.put_record("dataset_version", "v1", payload)
        assert first["id"] == second["id"]
        assert first["created_at"] == second["created_at"]

    def test_different_payload_same_id_raises(self, isolated_store: SqliteRunStore):
        isolated_store.put_record("dataset_version", "v1", {"version_id": "v1", "data": "a"})
        with pytest.raises(ValueError, match="immutable"):
            isolated_store.put_record("dataset_version", "v1", {"version_id": "v1", "data": "b"})

    def test_different_ids_independent(self, isolated_store: SqliteRunStore):
        isolated_store.put_record("dataset_version", "v1", {"data": "a"})
        isolated_store.put_record("dataset_version", "v2", {"data": "b"})
        assert isolated_store.get_record("dataset_version", "v1") is not None
        assert isolated_store.get_record("dataset_version", "v2") is not None

    def test_no_cross_test_leakage(self, isolated_store: SqliteRunStore):
        """Each test starts with an empty database."""
        assert isolated_store.get_record("dataset_version", "v1") is None


class TestStoreIsolation:
    """Each test must get its own SQLite file — no shared state."""

    def test_store_a(self, isolated_store: SqliteRunStore):
        isolated_store.put_record("test_type", "test_a", {"a": 1})
        assert isolated_store.get_record("test_type", "test_a") is not None
        assert isolated_store.get_record("test_type", "test_b") is None

    def test_store_b(self, isolated_store: SqliteRunStore):
        """Must not see test_a's data."""
        assert isolated_store.get_record("test_type", "test_a") is None
        isolated_store.put_record("test_type", "test_b", {"b": 2})
        assert isolated_store.get_record("test_type", "test_b") is not None
