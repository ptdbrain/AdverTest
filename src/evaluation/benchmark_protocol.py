"""Typed, immutable benchmark protocol contracts.

A BenchmarkProtocol locks every input that can affect metric reproducibility.
Once locked, any change creates a new protocol identity.  Checkpoint hash is
NOT part of protocol identity — it lives in BenchmarkRunManifest — so B0/R1/R2
can run the same locked protocol.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.hashing import stable_digest
from src.core.types import Task

ProtocolState = Literal["DRAFT", "VALIDATED", "LOCKED"]


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BenchmarkProtocol(_FrozenContract):
    """Immutable specification of what a benchmark measures."""

    protocol_id: str
    state: ProtocolState = "DRAFT"
    dataset_version_id: str
    model_version_id: str
    task: Task
    recipe: dict[str, Any]
    recipe_hash: str
    sample_ids: tuple[str, ...] = ()
    sample_hashes: tuple[str, ...] = ()
    dataset_hash: str = ""
    preprocessing_version: str = "default"
    thresholds: dict[str, float] = Field(default_factory=dict)
    metric_versions: dict[str, str] = Field(default_factory=dict)
    seed: int = Field(default=20260730, ge=0)
    prompt_protocol: str | None = None
    class_mapping_version: str = "1.0.0"
    bootstrap_config: dict[str, Any] = Field(default_factory=dict)
    schema_version: Literal["1.0.0"] = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.state == "LOCKED" and not self.sample_ids:
            raise ValueError("locked protocol must have sample_ids")
        if len(self.sample_ids) != len(self.sample_hashes) and self.sample_hashes:
            raise ValueError("sample_ids and sample_hashes must have equal length")
        return self

    def identity_hash(self) -> str:
        """Content-addressable identity: anything that affects metrics."""
        identity_payload = {
            "dataset_version_id": self.dataset_version_id,
            "task": self.task,
            "recipe_hash": self.recipe_hash,
            "sample_ids": self.sample_ids,
            "sample_hashes": self.sample_hashes,
            "dataset_hash": self.dataset_hash,
            "preprocessing_version": self.preprocessing_version,
            "thresholds": self.thresholds,
            "metric_versions": self.metric_versions,
            "seed": self.seed,
            "prompt_protocol": self.prompt_protocol,
            "class_mapping_version": self.class_mapping_version,
            "bootstrap_config": self.bootstrap_config,
        }
        return stable_digest(identity_payload, length=40)


class BenchmarkRunManifest(_FrozenContract):
    """Per-run provenance — checkpoint varies while protocol stays locked."""

    run_id: str
    protocol_id: str
    checkpoint_hash: str
    environment_versions: dict[str, str] = Field(default_factory=dict)
    sample_order: tuple[str, ...] = ()
    source_hashes: tuple[str, ...] = ()
    gt_hashes: tuple[str, ...] = ()
    schema_version: Literal["1.0.0"] = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_protocol_transition(
    current: ProtocolState,
    target: ProtocolState,
) -> bool:
    """Enforce DRAFT → VALIDATED → LOCKED; no backward transitions."""
    allowed = {
        "DRAFT": {"VALIDATED"},
        "VALIDATED": {"LOCKED"},
        "LOCKED": set(),
    }
    return target in allowed.get(current, set())
