"""Run report: the shape every UI, export, and CI gate reads.

One row per ``(attack, severity)`` cell, plus the clean baseline. Degradation
``D(c, s) = (AP_clean - AP(c, s)) / AP_clean`` is the headline number of plan §3
metric 2; the remaining aggregates are listed as open slots in
:mod:`src.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.types import AttackGroup


@dataclass(frozen=True, slots=True)
class CellResult:
    """Result for one attack at one severity."""

    attack: str
    group: AttackGroup
    severity: int
    ap: float
    n_samples: int
    seconds: float = 0.0
    cache_hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "attack": self.attack,
            "group": self.group,
            "severity": self.severity,
            "ap": round(self.ap, 4),
            "n_samples": self.n_samples,
            "seconds": round(self.seconds, 3),
            "cache_hits": self.cache_hits,
        }


@dataclass(frozen=True, slots=True)
class SkippedAttack:
    """An attack the runner could not apply, with the reason (never silent)."""

    attack: str
    reason: str


@dataclass
class RunReport:
    """Everything one test run produced."""

    run_id: str
    model: str
    model_version: str
    dataset: str
    n_samples: int
    ap_clean: float
    cells: list[CellResult] = field(default_factory=list)
    skipped: list[SkippedAttack] = field(default_factory=list)
    seconds: float = 0.0
    #: Never remove: this platform evaluates in simulation only (plan §7).
    simulation_only: bool = True

    def degradation(self, cell: CellResult) -> float:
        """``D(c, s)`` as a fraction in ``[0, 1]``; 0.0 when the baseline is 0."""
        if self.ap_clean <= 0.0:
            return 0.0
        return max(0.0, (self.ap_clean - cell.ap) / self.ap_clean)

    def heatmap(self) -> dict[str, dict[int, float]]:
        """``{attack: {severity: D(c, s)}}`` — rows and columns of the report grid."""
        grid: dict[str, dict[int, float]] = {}
        for cell in self.cells:
            grid.setdefault(cell.attack, {})[cell.severity] = round(self.degradation(cell), 4)
        return grid

    def worst_cases(self, top_n: int = 5) -> list[dict[str, Any]]:
        """Cells with the largest degradation — the "what breaks the model" list."""
        ranked = sorted(self.cells, key=lambda cell: cell.ap)
        return [
            {**cell.as_dict(), "degradation": round(self.degradation(cell), 4)}
            for cell in ranked[:top_n]
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "model_version": self.model_version,
            "dataset": self.dataset,
            "n_samples": self.n_samples,
            "ap_clean": round(self.ap_clean, 4),
            "cells": [
                {**cell.as_dict(), "degradation": round(self.degradation(cell), 4)}
                for cell in self.cells
            ],
            "heatmap": self.heatmap(),
            "worst_cases": self.worst_cases(),
            "skipped": [{"attack": item.attack, "reason": item.reason} for item in self.skipped],
            "seconds": round(self.seconds, 3),
            "simulation_only": True,
        }
