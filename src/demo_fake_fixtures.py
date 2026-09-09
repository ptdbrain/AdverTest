"""Validated, repository-shipped fixtures for the public fake demo workspace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.pipeline.runner import RunConfig

_EXPECTED_SESSIONS = ("DEMO-ROBUST-001", "DEMO-SEG-002", "DEMO-REVIEW-003")
_ARTIFACT_FIELDS = (
    "clean_image_path",
    "attacked_image_path",
    "clean_prediction_path",
    "attacked_prediction_path",
)


@dataclass(frozen=True, slots=True)
class FakeFixture:
    fixture_id: str
    session_id: str
    project_id: str
    baseline_run_id: str
    defence_run_id: str
    defence_checkpoint_id: str
    task_id: str
    task_name: str
    model_id: str
    model_family_id: str
    model_name: str
    dataset_id: str
    dataset_name: str
    report: dict[str, Any]
    defence_report: dict[str, Any]
    review: dict[str, Any]
    sample_reviews: tuple[dict[str, Any], ...]
    assets: tuple[Path, ...]

    def baseline_config(self, project_id: str | None = None) -> RunConfig:
        cells = self.report.get("cells", [])
        return RunConfig(
            model=self.model_name,
            project_id=project_id or self.project_id,
            model_family_id=self.model_family_id,
            checkpoint_id=self.model_id,
            model_version_id=self.model_id,
            task_id=self.task_id,
            dataset=self.dataset_id,
            dataset_params={"split": "val", "anonymize": "required", "fixture": self.fixture_id},
            attacks=[str(cell["attack"]) for cell in cells],
            severities=[int(cell["severity"]) for cell in cells],
            limit=self.report.get("n_samples", len(self.report.get("sample_results", []))) or 1,
            seed=20260906,
            adapter_params={"demo_fixture_id": self.fixture_id},
            execution_mode="benchmark",
        )

    def defence_config(self, project_id: str | None = None) -> RunConfig:
        return self.baseline_config(project_id).model_copy(
            update={
                "model": self.model_name,
                "checkpoint_id": self.defence_checkpoint_id,
                "model_version_id": self.defence_checkpoint_id,
                "adapter_params": {"demo_fixture_id": self.fixture_id, "defence": True},
            }
        )


def load_fake_fixture_catalog(data_root: str) -> list[FakeFixture]:
    """Load and validate all shipped fake fixtures without performing writes."""

    fixture_root = (Path(data_root).expanduser().resolve() / "demo" / "fake-sessions").resolve()
    manifest_path = fixture_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("DEMO_FIXTURE_MANIFEST_MISSING")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("DEMO_FIXTURE_MANIFEST_INVALID") from exc

    entries = manifest.get("fixtures") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or len(entries) != len(_EXPECTED_SESSIONS):
        raise ValueError("DEMO_FIXTURE_COUNT_INVALID")

    fixtures: list[FakeFixture] = []
    seen_sessions: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("DEMO_FIXTURE_ENTRY_INVALID")
        if entry.get("session_id") in seen_sessions:
            raise ValueError("DEMO_FIXTURE_SESSION_DUPLICATE")
        seen_sessions.add(str(entry.get("session_id")))
        report = _load_json(fixture_root, entry.get("report"), "DEMO_FIXTURE_REPORT_INVALID")
        defence_report = _load_json(
            fixture_root, entry.get("defence_report"), "DEMO_FIXTURE_DEFENCE_REPORT_INVALID"
        )
        assets = tuple(_resolve_asset(fixture_root, value) for value in entry.get("assets", []))
        report = _resolve_report_paths(report, fixture_root)
        defence_report = _resolve_report_paths(defence_report, fixture_root)
        _validate_report(report, str(entry.get("baseline_run_id")))
        _validate_report(defence_report, str(entry.get("defence_run_id")))
        fixtures.append(
            FakeFixture(
                fixture_id=str(entry["fixture_id"]),
                session_id=str(entry["session_id"]),
                project_id=str(entry["project_id"]),
                baseline_run_id=str(entry["baseline_run_id"]),
                defence_run_id=str(entry["defence_run_id"]),
                defence_checkpoint_id=str(entry["defence_checkpoint_id"]),
                task_id=str(entry["task_id"]),
                task_name=str(entry["task_name"]),
                model_id=str(entry["model_id"]),
                model_family_id=str(entry["model_family_id"]),
                model_name=str(entry["model_name"]),
                dataset_id=str(entry["dataset_id"]),
                dataset_name=str(entry["dataset_name"]),
                report=report,
                defence_report=defence_report,
                review=dict(entry.get("review", {})),
                sample_reviews=tuple(dict(item) for item in entry.get("sample_reviews", [])),
                assets=assets,
            )
        )

    if tuple(item.session_id for item in fixtures) != _EXPECTED_SESSIONS:
        raise ValueError("DEMO_FIXTURE_SESSION_IDS_INVALID")
    return fixtures


def _load_json(root: Path, relative: object, error_code: str) -> dict[str, Any]:
    if not isinstance(relative, str):
        raise ValueError(error_code)
    path = _resolve_path(root, relative, error_code)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(error_code) from exc
    if not isinstance(payload, dict):
        raise ValueError(error_code)
    return payload


def _resolve_asset(root: Path, relative: object) -> Path:
    path = _resolve_path(root, relative, "DEMO_FIXTURE_ASSET_INVALID")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError) as exc:
        raise ValueError("DEMO_FIXTURE_ASSET_INVALID") from exc
    if path.stat().st_size <= 0:
        raise ValueError("DEMO_FIXTURE_ASSET_INVALID")
    return path


def _resolve_path(root: Path, relative: object, error_code: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(error_code)
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("DEMO_FIXTURE_ASSET_ESCAPE")
    if not candidate.is_file():
        raise ValueError(error_code)
    return candidate


def _resolve_report_paths(report: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized = json.loads(json.dumps(report))
    for sample in [*normalized.get("sample_results", []), *normalized.get("worst_cases", [])]:
        if not isinstance(sample, dict):
            raise ValueError("DEMO_FIXTURE_SAMPLE_INVALID")
        for field in _ARTIFACT_FIELDS:
            if field in sample:
                sample[field] = str(_resolve_asset(root, sample[field]))
    return normalized


def _validate_report(report: dict[str, Any], expected_run_id: str) -> None:
    required = {"run_id", "model", "model_version", "dataset", "n_samples", "ap_clean", "cells", "sample_results"}
    if not required.issubset(report) or report.get("run_id") != expected_run_id:
        raise ValueError("DEMO_FIXTURE_REPORT_CONTRACT_INVALID")
    if report.get("simulation_only") is not True or (report.get("provenance") or {}).get("demo_fixture") is not True:
        raise ValueError("DEMO_FIXTURE_TRUTH_MARKER_MISSING")
