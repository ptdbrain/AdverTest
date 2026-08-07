"""Content-addressed hard-example artifacts with enforced usage permissions."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.core.hashing import array_digest
from src.core.objectives import AttackObjective
from src.core.types import Task
from src.evaluation.contracts import MetricEnvelope

AllowedUse = Literal["training", "benchmark", "review"]


class HardExampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    artifact_id: str
    task: Task
    source_sample_id: str
    source_hash: str
    model_id: str
    model_version: str
    attack_name: str
    attack_version: str
    attack_family: str
    protocol_id: str
    protocol_version: str
    objective: AttackObjective
    parameters: dict[str, Any] = Field(default_factory=dict)
    seeds: tuple[int, ...]
    before_metrics: tuple[MetricEnvelope, ...]
    after_metrics: tuple[MetricEnvelope, ...]
    failure_reason: str
    affected_instances: tuple[str, ...] = ()
    class_label: str | None = None
    object_size_bucket: str | None = None
    severity: int = Field(ge=0, le=5)
    artifact_hash: str
    allowed_uses: tuple[AllowedUse, ...]
    locked_test: bool = False
    provenance: dict[str, Any] = Field(default_factory=dict)
    schema_version: Literal["1.0.0"] = "1.0.0"


class HardExampleBank:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self.root / "index.sqlite3",
            timeout=30,
            check_same_thread=False,
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS hard_examples "
            "(artifact_id TEXT PRIMARY KEY, artifact_hash TEXT NOT NULL, "
            "payload TEXT NOT NULL)"
        )

    def put(self, record: HardExampleRecord, artifact: np.ndarray) -> None:
        _validate_record(record)
        if not isinstance(artifact, np.ndarray):
            raise TypeError("hard-example artifact must be a numpy array")
        actual_hash = array_digest(artifact, length=64)
        if actual_hash != record.artifact_hash:
            raise ValueError(
                f"artifact hash mismatch: {actual_hash} != {record.artifact_hash}"
            )
        destination = self._object_path(record.artifact_hash)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = np.load(destination, allow_pickle=False)
            if array_digest(existing, length=64) != record.artifact_hash:
                raise ValueError("content-addressed artifact hash mismatch on disk")
        else:
            temporary = destination.with_name(f".{destination.name}.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, artifact, allow_pickle=False)
            os.replace(temporary, destination)
        existing_row = self._connection.execute(
            "SELECT payload FROM hard_examples WHERE artifact_id=?",
            (record.artifact_id,),
        ).fetchone()
        serialized = record.model_dump_json()
        if existing_row is not None and existing_row[0] != serialized:
            raise ValueError(
                f"artifact_id {record.artifact_id!r} already has different provenance"
            )
        self._connection.execute(
            "INSERT OR REPLACE INTO hard_examples"
            "(artifact_id, artifact_hash, payload) VALUES (?, ?, ?)",
            (record.artifact_id, record.artifact_hash, serialized),
        )
        self._connection.commit()

    def get(
        self,
        artifact_id: str,
        *,
        intended_use: AllowedUse,
    ) -> tuple[HardExampleRecord, np.ndarray]:
        row = self._connection.execute(
            "SELECT artifact_hash, payload FROM hard_examples WHERE artifact_id=?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown hard example: {artifact_id!r}")
        record = HardExampleRecord.model_validate_json(row[1])
        if intended_use not in record.allowed_uses:
            raise PermissionError(
                f"hard example {artifact_id!r} is not allowed for {intended_use}"
            )
        artifact = np.load(self._object_path(row[0]), allow_pickle=False)
        if array_digest(artifact, length=64) != record.artifact_hash:
            raise ValueError(f"stored hard example {artifact_id!r} failed hash validation")
        return record, np.ascontiguousarray(artifact)

    def query(
        self,
        *,
        intended_use: AllowedUse,
        task: Task | None = None,
        attack_family: str | None = None,
        failure_reason: str | None = None,
        class_label: str | None = None,
        object_size_bucket: str | None = None,
        severity_min: int | None = None,
        severity_max: int | None = None,
    ) -> tuple[HardExampleRecord, ...]:
        rows = self._connection.execute(
            "SELECT payload FROM hard_examples ORDER BY artifact_id"
        ).fetchall()
        records = [
            HardExampleRecord.model_validate_json(row[0]) for row in rows
        ]
        return tuple(
            record
            for record in records
            if intended_use in record.allowed_uses
            and (task is None or record.task == task)
            and (
                attack_family is None
                or record.attack_family == attack_family
            )
            and (
                failure_reason is None
                or record.failure_reason == failure_reason
            )
            and (class_label is None or record.class_label == class_label)
            and (
                object_size_bucket is None
                or record.object_size_bucket == object_size_bucket
            )
            and (severity_min is None or record.severity >= severity_min)
            and (severity_max is None or record.severity <= severity_max)
        )

    def _object_path(self, artifact_hash: str) -> Path:
        return self.objects / artifact_hash[:2] / f"{artifact_hash}.npy"


def _validate_record(record: HardExampleRecord) -> None:
    required_provenance = {"dataset_version_id", "recipe_hash"}
    if not required_provenance.issubset(record.provenance):
        raise ValueError(
            "hard example provenance requires dataset_version_id and recipe_hash"
        )
    if not record.seeds:
        raise ValueError("hard example requires at least one seed")
    if record.locked_test and "training" in record.allowed_uses:
        raise ValueError("locked-test hard examples cannot be allowed for training")
    if not record.allowed_uses:
        raise ValueError("hard example requires at least one allowed use")
