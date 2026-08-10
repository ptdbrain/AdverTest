"""Durable SQLite state for background product workflows."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})


def _now() -> str:
    return datetime.now(UTC).isoformat()


class WorkflowJobStore:
    """Small, durable event store for workflows that must survive API restarts."""

    def __init__(self, database_url: str) -> None:
        self.path = _sqlite_path(database_url)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workflow_jobs (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS workflow_events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS retraining_backlogs (
                    backlog_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retraining_backlog_items (
                    backlog_id TEXT NOT NULL,
                    failure_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (backlog_id, failure_id)
                );
                """
            )

    def create_backlog(self, name: str) -> dict[str, Any]:
        backlog_id = f"backlog-{uuid.uuid4().hex}"
        now = _now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO retraining_backlogs(backlog_id, name, status, created_at, updated_at) VALUES (?, ?, 'DRAFT', ?, ?)",
                (backlog_id, name, now, now),
            )
        return self.get_backlog(backlog_id)  # type: ignore[return-value]

    def get_backlog(self, backlog_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM retraining_backlogs WHERE backlog_id=?", (backlog_id,)
            ).fetchone()
            if row is None:
                return None
            items = connection.execute(
                "SELECT failure_id FROM retraining_backlog_items WHERE backlog_id=? ORDER BY failure_id", (backlog_id,)
            ).fetchall()
        return {"id": row["backlog_id"], "name": row["name"], "status": row["status"], "failure_ids": [item["failure_id"] for item in items]}

    def add_backlog_item(self, backlog_id: str, failure_id: str) -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT status FROM retraining_backlogs WHERE backlog_id=?", (backlog_id,)).fetchone()
            if row is None:
                raise KeyError("BACKLOG_UNKNOWN")
            if row["status"] != "DRAFT":
                raise ValueError("BACKLOG_NOT_DRAFT")
            try:
                connection.execute(
                    "INSERT INTO retraining_backlog_items(backlog_id, failure_id, created_at) VALUES (?, ?, ?)",
                    (backlog_id, failure_id, _now()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("BACKLOG_ITEM_DUPLICATE") from exc
            connection.execute("UPDATE retraining_backlogs SET updated_at=? WHERE backlog_id=?", (_now(), backlog_id))
        return self.get_backlog(backlog_id)  # type: ignore[return-value]

    def approve_backlog(self, backlog_id: str) -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT status FROM retraining_backlogs WHERE backlog_id=?", (backlog_id,)).fetchone()
            if row is None:
                raise KeyError("BACKLOG_UNKNOWN")
            if row["status"] != "DRAFT":
                raise ValueError("BACKLOG_NOT_DRAFT")
            count = connection.execute("SELECT COUNT(*) AS count FROM retraining_backlog_items WHERE backlog_id=?", (backlog_id,)).fetchone()["count"]
            if count == 0:
                raise ValueError("BACKLOG_EMPTY")
            connection.execute("UPDATE retraining_backlogs SET status='APPROVED', updated_at=? WHERE backlog_id=?", (_now(), backlog_id))
        return self.get_backlog(backlog_id)  # type: ignore[return-value]

    def create_job(self, job_type: str, request: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        now = _now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT INTO workflow_jobs
                   (job_id, job_type, request_json, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'QUEUED', ?, ?)""",
                (job_id, job_type, _json(request), now, now),
            )
            self._append_event(connection, job_id, "QUEUED", {"progress_ratio": 0.0})
        return job_id

    def append_event(
        self, job_id: str, state: str, payload: dict[str, Any], *, update_status: bool = True
    ) -> bool:
        with self._lock, self._connection() as connection:
            if not self._exists(connection, job_id):
                return False
            if update_status:
                connection.execute(
                    "UPDATE workflow_jobs SET status=?, updated_at=? WHERE job_id=?",
                    (state, _now(), job_id),
                )
            self._append_event(connection, job_id, state, payload)
        return True

    def request_cancel(self, job_id: str) -> bool:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT status FROM workflow_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                return False
            if row["status"] not in TERMINAL_STATES:
                connection.execute(
                    "UPDATE workflow_jobs SET cancel_requested=1, updated_at=? WHERE job_id=?",
                    (_now(), job_id),
                )
                self._append_event(connection, job_id, "CANCEL_REQUESTED", {})
        return True

    def complete_job(self, job_id: str, result: dict[str, Any]) -> bool:
        with self._lock, self._connection() as connection:
            if not self._exists(connection, job_id):
                return False
            now = _now()
            connection.execute(
                "UPDATE workflow_jobs SET status='COMPLETED', result_json=?, error=NULL, updated_at=? WHERE job_id=?",
                (_json(result), now, job_id),
            )
            self._append_checkpoint(connection, job_id, result)
            self._append_event(connection, job_id, "COMPLETED", {"progress_ratio": 1.0})
        return True

    def fail_job(self, job_id: str, error: str, *, cancelled: bool = False) -> bool:
        state = "CANCELLED" if cancelled else "FAILED"
        with self._lock, self._connection() as connection:
            if not self._exists(connection, job_id):
                return False
            connection.execute(
                "UPDATE workflow_jobs SET status=?, error=?, updated_at=? WHERE job_id=?",
                (state, error, _now(), job_id),
            )
            self._append_event(connection, job_id, state, {"error": error})
        return True

    def checkpoint(self, job_id: str, payload: dict[str, Any]) -> bool:
        with self._lock, self._connection() as connection:
            if not self._exists(connection, job_id):
                return False
            self._append_checkpoint(connection, job_id, payload)
        return True

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return _job(row) if row else None

    def events(self, job_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM workflow_events WHERE job_id=? ORDER BY sequence", (job_id,)
            ).fetchall()
        return [_event(row) for row in rows]

    def checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM workflow_checkpoints WHERE job_id=? ORDER BY sequence", (job_id,)
            ).fetchall()
        return [_checkpoint(row) for row in rows]

    def cancel_requested(self, job_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM workflow_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def recoverable(self, job_type: str) -> list[dict[str, Any]]:
        """Return and requeue interrupted jobs of one workflow type."""
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM workflow_jobs
                   WHERE job_type=? AND status NOT IN ('COMPLETED', 'FAILED', 'CANCELLED')""",
                (job_type,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE workflow_jobs SET status='QUEUED', updated_at=? WHERE job_id=?",
                    (_now(), row["job_id"]),
                )
                self._append_event(connection, row["job_id"], "QUEUED", {"recovered": True})
        return [_job(row) for row in rows]

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _exists(connection: sqlite3.Connection, job_id: str) -> bool:
        return connection.execute(
            "SELECT 1 FROM workflow_jobs WHERE job_id=?", (job_id,)
        ).fetchone() is not None

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection, job_id: str, state: str, payload: dict[str, Any]
    ) -> None:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 AS value FROM workflow_events WHERE job_id=?",
            (job_id,),
        ).fetchone()["value"]
        connection.execute(
            "INSERT INTO workflow_events(job_id, sequence, state, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, sequence, state, _json(payload), _now()),
        )

    @staticmethod
    def _append_checkpoint(
        connection: sqlite3.Connection, job_id: str, payload: dict[str, Any]
    ) -> None:
        sequence = connection.execute(
            "SELECT COALESCE(MAX(sequence), -1) + 1 AS value FROM workflow_checkpoints WHERE job_id=?",
            (job_id,),
        ).fetchone()["value"]
        connection.execute(
            "INSERT INTO workflow_checkpoints(job_id, sequence, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, sequence, _json(payload), _now()),
        )


def _sqlite_path(database_url: str) -> Path:
    if not database_url.startswith("sqlite:///"):
        raise ValueError("workflow job store requires a sqlite:/// database_url")
    return Path(database_url.removeprefix("sqlite:///"))


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _job(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["job_id"],
        "job_type": row["job_type"],
        "request": json.loads(row["request_json"]),
        "status": row["status"],
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": row["error"],
        "cancel_requested": bool(row["cancel_requested"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "sequence": row["sequence"],
        "state": row["state"],
        "payload": json.loads(row["payload_json"]),
        "created_at": row["created_at"],
    }


def _checkpoint(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "sequence": row["sequence"],
        "payload": json.loads(row["payload_json"]),
        "created_at": row["created_at"],
    }
