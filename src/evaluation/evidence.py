"""Evidence classification shared by reports, exports, promotion, and UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

EvidenceStatus = Literal["VERIFIED", "NOT_ELIGIBLE", "WAITING_FOR_GPU_VALIDATION", "INVALID"]

REQUIRED_EVIDENCE: tuple[str, ...] = (
    "dataset_version_id",
    "split_manifest_hash",
    "ground_truth_hash",
    "checkpoint_sha256",
    "config_sha256",
    "artifact_hashes",
    "metric_protocol",
)


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """Immutable decision and explanation for one report's scientific evidence."""

    status: EvidenceStatus
    missing: tuple[str, ...]
    action: str

    @property
    def promotion_eligible(self) -> bool:
        return self.status == "VERIFIED"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "missing": list(self.missing),
            "action": self.action,
            "promotion_eligible": self.promotion_eligible,
        }


def evaluate_evidence(
    provenance: Mapping[str, Any],
    *,
    simulation_only: bool,
) -> EvidenceSnapshot:
    """Classify whether supplied provenance can support a benchmark conclusion."""
    missing = tuple(key for key in REQUIRED_EVIDENCE if not provenance.get(key))
    runtime_status = str(provenance.get("runtime_validation_status", "")).upper()
    if runtime_status == "WAITING_FOR_GPU_VALIDATION":
        return EvidenceSnapshot(
            status="WAITING_FOR_GPU_VALIDATION",
            missing=missing,
            action="Run compatible CUDA inference and record finite non-empty predictions.",
        )
    if simulation_only:
        return EvidenceSnapshot(
            status="NOT_ELIGIBLE",
            missing=("simulation_only", *missing),
            action="Run the locked protocol against ground truth before promotion or benchmark export.",
        )
    if missing:
        return EvidenceSnapshot(
            status="NOT_ELIGIBLE",
            missing=missing,
            action="Attach every missing evidence item before promotion or benchmark export.",
        )
    return EvidenceSnapshot(status="VERIFIED", missing=(), action="Evidence is complete for the declared protocol.")
