"""Immutable benchmark protocol identity and lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.hashing import stable_digest

ProtocolStatus = Literal["DRAFT", "VALIDATED", "LOCKED", "RETIRED"]


class BenchmarkProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str
    dataset_version_id: str
    sample_ids: tuple[str, ...]
    sample_hashes: dict[str, str]
    ground_truth_hashes: dict[str, str]
    recipe_hashes: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    preprocessing_versions: dict[str, str] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    prompt_protocol: str | None = None
    metric_versions: dict[str, str] = Field(default_factory=dict)
    bootstrap_iterations: int = Field(default=1_000, ge=1)
    bootstrap_seed: int = Field(default=195, ge=0)
    environment: dict[str, str] = Field(default_factory=dict)
    framework_versions: dict[str, str] = Field(default_factory=dict)
    class_mapping_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    status: ProtocolStatus = "DRAFT"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    protocol_id: str = ""

    @model_validator(mode="after")
    def derive_identity(self) -> Self:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("protocol sample_ids must be unique")
        if set(self.sample_hashes) != set(self.sample_ids):
            raise ValueError("sample_hashes must cover exactly sample_ids")
        if set(self.ground_truth_hashes) != set(self.sample_ids):
            raise ValueError("ground_truth_hashes must cover exactly sample_ids")
        payload = self.model_dump(
            mode="json",
            exclude={"created_at", "protocol_id", "status"},
        )
        identity = f"protocol-{stable_digest(payload, length=32)}"
        if self.protocol_id and self.protocol_id != identity:
            raise ValueError("protocol_id does not match immutable content")
        object.__setattr__(self, "protocol_id", identity)
        return self

    @classmethod
    def minimal(
        cls,
        *,
        name: str,
        dataset_version_id: str,
        sample_ids: tuple[str, ...],
    ) -> BenchmarkProtocol:
        return cls(
            name=name,
            dataset_version_id=dataset_version_id,
            sample_ids=sample_ids,
            sample_hashes={sample_id: f"unverified:{sample_id}" for sample_id in sample_ids},
            ground_truth_hashes={sample_id: f"unverified:{sample_id}" for sample_id in sample_ids},
        )

    def transition(self, target: ProtocolStatus) -> BenchmarkProtocol:
        allowed = {
            "DRAFT": {"VALIDATED"},
            "VALIDATED": {"LOCKED"},
            "LOCKED": {"RETIRED"},
            "RETIRED": set(),
        }
        if target not in allowed[self.status]:
            raise ValueError(f"illegal protocol transition {self.status} -> {target}")
        return self.model_copy(update={"status": target})


class ProtocolValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    protocol_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
