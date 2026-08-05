"""Durable local job backend used by the API and replaceable in deployment.

SQLite is intentionally the default so a developer can run the complete API
without provisioning infrastructure. The repository interface separates it
from the worker; production can provide the same methods through PostgreSQL,
Redis and Celery without changing routes or runner semantics.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.pipeline import RunConfig, TestRunner
from src.pipeline.cache import SqliteCache
from src.pipeline.runner import RunCancelledError

TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
RUN_STATES = frozenset({"QUEUED", "PREPARING", "GENERATING", "INFERENCING", "EVALUATING", *TERMINAL_STATES})


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SqliteRunStore:
    def __init__(self, database_url: str) -> None:
        self.path = _sqlite_path(database_url)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS test_runs (
                    run_id TEXT PRIMARY KEY, config_json TEXT NOT NULL, status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0, detail_json TEXT NOT NULL DEFAULT '{}',
                    report_json TEXT, error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS test_run_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    state TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metric_results (
                    run_id TEXT NOT NULL, attack TEXT NOT NULL, severity INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL, PRIMARY KEY (run_id, attack, severity)
                );
                CREATE TABLE IF NOT EXISTS sample_results (
                    run_id TEXT NOT NULL, sample_id TEXT NOT NULL, attack TEXT NOT NULL,
                    severity INTEGER NOT NULL, payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sample_id, attack, severity)
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    attack TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    dataset TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    degradation REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    decision TEXT,
                    decision_note TEXT,
                    flagged_by TEXT NOT NULL DEFAULT 'system_auto',
                    resolved_by TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, config: RunConfig) -> str:
        run_id = uuid.uuid4().hex[:16]
        now = _now()
        with self._lock, self._connection() as connection:
            connection.execute(
                "INSERT INTO test_runs(run_id, config_json, status, created_at, updated_at) VALUES (?, ?, 'QUEUED', ?, ?)",
                (run_id, config.model_dump_json(), now, now),
            )
            self._event(connection, run_id, "QUEUED", {"progress": 0.0})
        return run_id

    def update(self, run_id: str, state: str, *, progress: float, detail: dict[str, Any] | None = None) -> None:
        if state not in RUN_STATES:
            raise ValueError(f"invalid run state {state!r}")
        payload = detail or {}
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE test_runs SET status=?, progress=?, detail_json=?, updated_at=? WHERE run_id=?",
                (state, min(1.0, max(0.0, progress)), json.dumps(payload), _now(), run_id),
            )
            self._event(connection, run_id, state, {"progress": progress, **payload})

    def complete(self, run_id: str, report: dict[str, Any]) -> None:
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE test_runs SET status='COMPLETED', progress=1, report_json=?, updated_at=? WHERE run_id=?",
                (json.dumps(report), _now(), run_id),
            )
            for cell in report.get("cells", []):
                connection.execute(
                    "INSERT OR REPLACE INTO metric_results VALUES (?, ?, ?, ?)",
                    (run_id, cell["attack"], cell["severity"], json.dumps(cell.get("metrics", {}))),
                )
            for sample in report.get("sample_results", []):
                connection.execute(
                    "INSERT OR REPLACE INTO sample_results VALUES (?, ?, ?, ?, ?)",
                    (run_id, sample["sample_id"], sample["attack"], sample["severity"], json.dumps(sample)),
                )
            self._event(connection, run_id, "COMPLETED", {"progress": 1.0})

    def checkpoint(self, run_id: str, report: dict[str, Any]) -> None:
        """Persist completed cells so a worker crash never erases evidence."""
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE test_runs SET report_json=?, updated_at=? WHERE run_id=?",
                (json.dumps(report), _now(), run_id),
            )
            for cell in report.get("cells", []):
                connection.execute(
                    "INSERT OR REPLACE INTO metric_results VALUES (?, ?, ?, ?)",
                    (run_id, cell["attack"], cell["severity"], json.dumps(cell.get("metrics", {}))),
                )
            for sample in report.get("sample_results", []):
                connection.execute(
                    "INSERT OR REPLACE INTO sample_results VALUES (?, ?, ?, ?, ?)",
                    (run_id, sample["sample_id"], sample["attack"], sample["severity"], json.dumps(sample)),
                )
            self._event(connection, run_id, "CHECKPOINT", {"cells": len(report.get("cells", []))})

    def fail(self, run_id: str, error: str, *, cancelled: bool = False) -> None:
        state = "CANCELLED" if cancelled else "FAILED"
        with self._lock, self._connection() as connection:
            connection.execute(
                "UPDATE test_runs SET status=?, error=?, updated_at=? WHERE run_id=?",
                (state, error, _now(), run_id),
            )
            self._event(connection, run_id, state, {"error": error})

    def request_cancel(self, run_id: str) -> bool:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT status FROM test_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return False
            if row["status"] in TERMINAL_STATES:
                return True
            connection.execute("UPDATE test_runs SET cancel_requested=1, updated_at=? WHERE run_id=?", (_now(), run_id))
            self._event(connection, run_id, "CANCEL_REQUESTED", {})
        return True

    def cancel_requested(self, run_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute("SELECT cancel_requested FROM test_runs WHERE run_id=?", (run_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM test_runs WHERE run_id=?", (run_id,)).fetchone()
        return _row_payload(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM test_runs ORDER BY created_at DESC").fetchall()
        return [_row_payload(row) for row in rows]

    def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM test_run_events WHERE run_id=? AND event_id>? ORDER BY event_id", (run_id, after)
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "state": row["state"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def recoverable(self) -> list[tuple[str, RunConfig]]:
        """Requeue interrupted jobs after a backend restart.

        Completed cell checkpoints and the persistent prediction cache remain
        available; the runner can safely repeat an incomplete cell.
        """
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT run_id, config_json FROM test_runs WHERE status NOT IN ('COMPLETED', 'FAILED', 'CANCELLED')"
            ).fetchall()
            for row in rows:
                connection.execute(
                    "UPDATE test_runs SET status='QUEUED', updated_at=? WHERE run_id=?",
                    (_now(), row["run_id"]),
                )
                self._event(connection, row["run_id"], "QUEUED", {"recovered": True})
        return [(row["run_id"], RunConfig.model_validate_json(row["config_json"])) for row in rows]

    @staticmethod
    def _event(connection: sqlite3.Connection, run_id: str, state: str, payload: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO test_run_events(run_id, state, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (run_id, state, json.dumps(payload), _now()),
        )

    # ---- Review CRUD ----

    def auto_flag_reviews(self, run_id: str, *, threshold: float = 30.0) -> list[str]:
        """Create review items for cells exceeding the degradation threshold."""
        item = self.get(run_id)
        if item is None or item["report"] is None:
            return []
        report = item["report"]
        config = json.loads(
            self._connection().execute(
                "SELECT config_json FROM test_runs WHERE run_id=?", (run_id,)
            ).fetchone()["config_json"]
        )
        created_ids: list[str] = []
        now = _now()
        with self._lock, self._connection() as connection:
            for cell in report.get("cells", []):
                if cell.get("degradation", 0) >= threshold:
                    review_id = f"REV-{uuid.uuid4().hex[:8]}"
                    connection.execute(
                        """INSERT OR IGNORE INTO reviews
                           (review_id, run_id, attack, severity, dataset, model, degradation, status, flagged_by, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', 'system_auto', ?, ?)""",
                        (
                            review_id, run_id, cell["attack"], cell["severity"],
                            report.get("dataset", config.get("dataset", "")),
                            report.get("model", config.get("model", "")),
                            cell["degradation"], now, now,
                        ),
                    )
                    created_ids.append(review_id)
        return created_ids

    def create_review(self, run_id: str, attack: str, severity: int, degradation: float,
                      dataset: str, model: str, flagged_by: str, notes: str = "") -> str:
        """Manually create a review item."""
        review_id = f"REV-{uuid.uuid4().hex[:8]}"
        now = _now()
        with self._lock, self._connection() as connection:
            connection.execute(
                """INSERT INTO reviews
                   (review_id, run_id, attack, severity, dataset, model, degradation, status, flagged_by, decision_note, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)""",
                (review_id, run_id, attack, severity, dataset, model, degradation, flagged_by, notes, now, now),
            )
        return review_id

    def list_reviews(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as connection:
            if status:
                rows = connection.execute(
                    "SELECT * FROM reviews WHERE status=? ORDER BY created_at DESC", (status,)
                ).fetchall()
            else:
                rows = connection.execute("SELECT * FROM reviews ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM reviews WHERE review_id=?", (review_id,)).fetchone()
        return dict(row) if row else None

    def resolve_review(self, review_id: str, decision: str, decision_note: str, resolved_by: str) -> bool:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT status FROM reviews WHERE review_id=?", (review_id,)).fetchone()
            if row is None:
                return False
            connection.execute(
                """UPDATE reviews SET status='RESOLVED', decision=?, decision_note=?, resolved_by=?, updated_at=?
                   WHERE review_id=?""",
                (decision, decision_note, resolved_by, _now(), review_id),
            )
        return True


class LocalRunWorker:
    """Bounded background worker. A GPU deployment replaces this with Celery."""

    def __init__(self, store: SqliteRunStore, *, max_workers: int = 1) -> None:
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="advertest-run")

    def enqueue(self, run_id: str, config: RunConfig) -> None:
        self.pool.submit(self._execute, run_id, config)

    def _execute(self, run_id: str, config: RunConfig) -> None:
        total_cells = max(1, len(config.attacks) * len(config.severities))

        def progress(state: str, detail: dict[str, Any]) -> None:
            if state == "PREPARING":
                fraction = 0.05
            elif state == "INFERENCING":
                fraction = 0.15
            elif state == "GENERATING":
                fraction = 0.25
            else:
                fraction = 0.95
            self.store.update(run_id, state, progress=fraction, detail=detail)

        try:
            config = config.model_copy(
                update={"evidence_dir": config.evidence_dir or str(self.store.path.parent / "artifacts" / run_id)}
            )
            runner = TestRunner(SqliteCache(str(self.store.path.parent / "prediction-cache.db")))

            def checkpoint(partial: Any) -> None:
                self.store.checkpoint(run_id, partial.as_dict())
                self.store.update(
                    run_id,
                    "GENERATING",
                    progress=0.25 + 0.65 * len(partial.cells) / total_cells,
                    detail={"completed_cells": len(partial.cells)},
                )

            report = runner.run(
                config,
                progress=progress,
                should_cancel=lambda: self.store.cancel_requested(run_id),
                checkpoint=checkpoint,
                run_id=run_id,
            )
            self.store.complete(run_id, report.as_dict())
        except RunCancelledError as exc:
            self.store.fail(run_id, str(exc), cancelled=True)
        except Exception as exc:  # job boundary: persist an actionable error
            self.store.fail(run_id, f"{type(exc).__name__}: {exc}")


def _sqlite_path(url: str) -> Path:
    if not url.startswith("sqlite:///"):
        raise ValueError("local job backend requires a sqlite:/// database_url")
    return Path(url.removeprefix("sqlite:///"))


def _row_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "status": row["status"],
        "progress": row["progress"],
        "detail": json.loads(row["detail_json"]),
        "report": json.loads(row["report_json"]) if row["report_json"] else None,
        "error": row["error"],
        "cancel_requested": bool(row["cancel_requested"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
