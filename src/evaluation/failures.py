"""Deterministic failure grouping for reproducible targeted sampling."""

from __future__ import annotations

from collections import defaultdict

from src.core.hashing import stable_digest
from src.evaluation.contracts import FailureCase, FailureCluster


class FailureGrouper:
    version = "1.0.0"

    def group(
        self,
        cases: tuple[FailureCase, ...],
    ) -> tuple[FailureCluster, ...]:
        grouped: dict[tuple[str, ...], list[FailureCase]] = defaultdict(list)
        for case in cases:
            metadata = case.metadata
            key = (
                str(metadata.get("task", "unknown")),
                str(metadata.get("failure_type", case.reason)),
                str(metadata.get("class_label", "unknown")),
                str(metadata.get("object_size_bucket", "unknown")),
                str(metadata.get("attack_family", "unknown")),
                _severity_band(int(metadata.get("severity", 0))),
            )
            grouped[key].append(case)
        clusters: list[FailureCluster] = []
        for key, members in sorted(grouped.items()):
            member_ids = tuple(sorted(case.case_id for case in members))
            allowed_sets = [
                set(case.metadata.get("allowed_uses", ("review",)))
                for case in members
            ]
            allowed = (
                set.intersection(*allowed_sets)
                if allowed_sets
                else {"review"}
            )
            cluster_id = f"failure-cluster-{stable_digest({'key': key, 'members': member_ids}, length=20)}"
            selection_allowed = "training" in allowed
            clusters.append(
                FailureCluster(
                    cluster_id=cluster_id,
                    member_ids=member_ids,
                    selection_allowed=selection_allowed,
                    allowed_uses=tuple(sorted(allowed)),
                    selection_reason=(
                        "all members permit training selection"
                        if selection_allowed
                        else "member permissions exclude training"
                    ),
                    method="deterministic-rules",
                    version=self.version,
                    metadata={
                        "task": key[0],
                        "failure_type": key[1],
                        "class_label": key[2],
                        "object_size_bucket": key[3],
                        "attack_family": key[4],
                        "severity_band": key[5],
                    },
                )
            )
        return tuple(clusters)


def _severity_band(severity: int) -> str:
    if severity <= 0:
        return "none"
    if severity <= 2:
        return "low"
    if severity == 3:
        return "medium"
    return "high"
