"""Database factory shared by the API process and background workers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.persistence.models import Base


class PlatformDatabase:
    """Owns an engine and exposes transaction-scoped sessions.

    ``create_schema`` is intended for local development and tests. Production
    runs Alembic as a release command before web and worker processes start.
    """

    def __init__(self, database_url: str) -> None:
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(database_url, future=True, pool_pre_ping=True, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self.engine, class_=Session, autoflush=False, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
