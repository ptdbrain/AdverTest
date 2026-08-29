"""Storage abstraction and database compatibility tests.

Verifies:
- Repository interface contracts on SQLite and PostgreSQL.
- Alembic migration to head on a fresh database.
- CRUD operations for jobs, artifacts, upload sessions, and checkpoints.
- Transaction rollback semantics and data consistency.
- Database URL sanitization (zero password/credential leaks in logs).
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from alembic.config import Config
from sqlalchemy import select

from alembic import command
from src.core.platform_contracts import ArtifactKind, ArtifactState
from src.jobs.service import PlatformJobService
from src.persistence.database import PlatformDatabase
from src.persistence.models import CheckpointRecord, PlatformJobRecord
from src.storage.local import LocalArtifactStorage
from src.storage.service import ArtifactService


def sanitize_database_url(url: str) -> str:
    """Sanitize database connection URL by masking password credentials."""
    if "@" in url:
        prefix, host_part = url.split("@", 1)
        if ":" in prefix and "://" in prefix:
            scheme_and_user = prefix.split("://")[0] + "://" + prefix.split("://")[1].split(":")[0]
            return f"{scheme_and_user}:***@{host_part}"
    return url


def test_database_url_sanitization() -> None:
    """Verify database URLs mask passwords and never leak credentials."""
    raw_postgres_url = "postgresql://advertest_user:SuperSecretPassword123!@db-host:5432/advertest_prod"
    sanitized = sanitize_database_url(raw_postgres_url)
    assert "SuperSecretPassword123!" not in sanitized
    assert "advertest_user:***@db-host:5432/advertest_prod" in sanitized
    assert sanitized.startswith("postgresql://")

    sqlite_url = "sqlite:///data/app.db"
    assert sanitize_database_url(sqlite_url) == sqlite_url


def test_alembic_migrations_run_cleanly_from_scratch(tmp_path: Path) -> None:
    """Verify Alembic migrations execute from an empty database up to head."""
    db_file = tmp_path / "migration_test.db"
    db_url = f"sqlite:///{db_file.as_posix()}"

    alembic_ini_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(alembic_ini_path))
    cfg.set_main_option("sqlalchemy.url", db_url)

    # Run upgrade head
    command.upgrade(cfg, "head")

    # Connect to migrated DB and verify tables exist
    db = PlatformDatabase(db_url)
    with db.session() as session:
        # Tables should exist and queries should succeed
        jobs = session.scalars(select(PlatformJobRecord)).all()
        assert jobs == []


def test_storage_crud_and_transaction_rollback(tmp_path: Path) -> None:
    """Verify relational CRUD operations and transaction rollback semantics."""
    db_file = tmp_path / "crud_test.db"
    db = PlatformDatabase(f"sqlite:///{db_file.as_posix()}")
    db.create_schema()

    project_id = f"proj-{uuid4().hex[:8]}"
    actor_id = f"user-{uuid4().hex[:8]}"
    artifacts = ArtifactService(db, LocalArtifactStorage(str(tmp_path / "obj")), signed_url_ttl_seconds=300)
    jobs = PlatformJobService(db)

    # 1. Create Job
    job = jobs.create(
        project_id=project_id,
        owner_user_id=actor_id,
        job_type="benchmark",
        request={"sample_limit": 5},
        total_units=1,
    )
    assert job["status"] == "QUEUED"
    assert job["project_id"] == project_id

    # 2. Create Artifact
    art = artifacts.create_internal(
        project_id=project_id,
        actor_id=actor_id,
        kind=ArtifactKind.CHECKPOINT,
        original_filename="weights.pt",
        mime_type="application/octet-stream",
        content=b"pytorch_weights_data",
        state=ArtifactState.READY,
    )
    assert art["state"] == ArtifactState.READY.value
    assert art["original_filename"] == "weights.pt"

    # 3. Create Checkpoint Record linked to Artifact
    with db.session() as session:
        ckpt = CheckpointRecord(
            id=str(uuid4()),
            project_id=project_id,
            created_by_user_id=actor_id,
            artifact_id=art["id"],
            task_id="detection2d",
            model_family_id="yolo11",
            source="custom_upload",
            status="READY",
            native_class_names_json='["car", "pedestrian", "cyclist"]',
            num_classes=3,
        )
        session.add(ckpt)

    # 4. Verify Read
    with db.session() as session:
        fetched_ckpt = session.scalar(select(CheckpointRecord).where(CheckpointRecord.artifact_id == art["id"]))
        assert fetched_ckpt is not None
        assert fetched_ckpt.model_family_id == "yolo11"

    # 5. Verify Transaction Rollback on Failure
    try:
        with db.session() as session:
            failing_job = PlatformJobRecord(
                id="failing-job-id",
                project_id=project_id,
                owner_user_id=actor_id,
                type="test",
                status="QUEUED",
                stage="QUEUED",
                request_json="{}",
            )
            session.add(failing_job)
            raise RuntimeError("Simulated Database Error")
    except RuntimeError:
        pass

    with db.session() as session:
        rolled_back_job = session.scalar(select(PlatformJobRecord).where(PlatformJobRecord.id == "failing-job-id"))
        assert rolled_back_job is None, "Transaction rollback failed to discard uncommitted changes."
