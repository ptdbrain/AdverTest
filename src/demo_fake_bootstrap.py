"""Idempotent bootstrap for the opt-in, static fake demo workspace."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from src.api import dependencies as api_dependencies
from src.api import platform_dependencies
from src.auth import dependencies as auth_dependencies
from src.config import get_settings
from src.demo_fake_fixtures import FakeFixture, load_fake_fixture_catalog
from src.persistence.models import ExperimentSessionRecord, ProjectMembershipRecord, ProjectRecord, SessionRunRecord

_BOOTSTRAP_LOCK = Lock()
_DEMO_DISPLAY_NAME = "Demo Engineer (Prepared Walkthrough)"


def ensure_fake_demo_engineer():
    settings = get_settings()
    return auth_dependencies.get_auth_service().ensure_seeded_account(
        user_id=settings.demo_fake_engineer_id,
        email=settings.demo_fake_engineer_email,
        password=settings.demo_fake_engineer_password,
        display_name=_DEMO_DISPLAY_NAME,
        role="ENGINEER",
    )


def ensure_fake_demo_workspace() -> list[FakeFixture]:
    """Seed the complete fake workspace once and return its fixture catalog."""

    settings = get_settings()
    if not getattr(settings, "demo_fake_sessions", False):
        return []
    with _BOOTSTRAP_LOCK:
        fixtures = load_fake_fixture_catalog(settings.data_root)
        if settings.run_execution_backend != "local":
            raise RuntimeError("DEMO_FAKE_SESSIONS_REQUIRES_LOCAL_RUN_BACKEND")
        user = ensure_fake_demo_engineer()
        database = platform_dependencies.get_platform_database()
        if settings.app_env != "production":
            database.create_schema()
        for fixture in fixtures:
            _ensure_project_and_session(database, fixture, user.id)
            store = api_dependencies.get_store()
            store.seed_completed(fixture.baseline_run_id, fixture.baseline_config(fixture.project_id), fixture.report)
            store.seed_completed(fixture.defence_run_id, fixture.defence_config(fixture.project_id), fixture.defence_report)
            store.seed_review(fixture.review)
            if fixture.sample_reviews:
                store.upsert_sample_reviews(fixture.baseline_run_id, list(fixture.sample_reviews))
            _ensure_session_run(database, fixture)
        return fixtures


def fixture_for_baseline(record: dict[str, Any]) -> FakeFixture | None:
    """Resolve a fixture marker from one stored run without doing inference."""

    marker = ((record.get("config") or {}).get("adapter_params") or {}).get("demo_fixture_id")
    if not marker:
        return None
    settings = get_settings()
    for fixture in load_fake_fixture_catalog(settings.data_root):
        if fixture.fixture_id == marker:
            return fixture
    raise ValueError("DEMO_FIXTURE_UNKNOWN")


def _ensure_project_and_session(database, fixture: FakeFixture, owner_user_id: str) -> None:
    with database.session() as db:
        project = db.get(ProjectRecord, fixture.project_id)
        if project is None:
            project = ProjectRecord(
                id=fixture.project_id,
                name=_project_name(fixture),
                description=_project_description(fixture),
                task_type=fixture.task_id,
                owner_user_id=owner_user_id,
            )
            db.add(project)
            db.flush()
        elif project.owner_user_id != owner_user_id or project.task_type != fixture.task_id:
            raise ValueError("DEMO_PROJECT_IDENTITY_CONFLICT")

        membership = (
            db.query(ProjectMembershipRecord)
            .filter_by(project_id=fixture.project_id, user_id=owner_user_id)
            .first()
        )
        if membership is None:
            db.add(
                ProjectMembershipRecord(
                    id=f"demo-membership-{fixture.fixture_id}",
                    project_id=fixture.project_id,
                    user_id=owner_user_id,
                    role="OWNER",
                    status="ACTIVE",
                )
            )

        session = db.get(ExperimentSessionRecord, fixture.session_id)
        if session is None:
            session = ExperimentSessionRecord(
                id=fixture.session_id,
                project_id=fixture.project_id,
                owner_user_id=owner_user_id,
                name=_session_name(fixture),
                description=_session_description(fixture),
                task_id=fixture.task_id,
                task_name=fixture.task_name,
                model_id=fixture.model_id,
                model_name=fixture.model_name,
                dataset_id=fixture.dataset_id,
                dataset_name=fixture.dataset_name,
                class_mapping_json=json.dumps(_class_mapping(fixture), ensure_ascii=False),
                status="active",
            )
            db.add(session)
        elif session.project_id != fixture.project_id or session.task_id != fixture.task_id:
            raise ValueError("DEMO_SESSION_IDENTITY_CONFLICT")


def _ensure_session_run(database, fixture: FakeFixture) -> None:
    report = fixture.report
    cell = (report.get("cells") or [{}])[0]
    sample = (report.get("sample_results") or [{}])[0]
    with database.session() as db:
        if db.get(SessionRunRecord, fixture.baseline_run_id) is not None:
            return
        clean_prediction = sample.get("clean_prediction") or {}
        attacked_prediction = sample.get("attacked_prediction") or {}
        clean_items = clean_prediction.get("boxes") or clean_prediction.get("instances") or []
        attacked_items = attacked_prediction.get("boxes") or attacked_prediction.get("instances") or []
        clean_conf = _mean_score(clean_items)
        attacked_conf = _mean_score(attacked_items)
        degradation = float(cell.get("degradation_percent", 0.0))
        db.add(
            SessionRunRecord(
                id=fixture.baseline_run_id,
                session_id=fixture.session_id,
                project_id=fixture.project_id,
                name=f"Lần 1: {fixture.report['cells'][0]['attack']}",
                timestamp=datetime.now(UTC).isoformat(),
                attack_type=str(cell.get("attack", "demo_fixture")),
                attack_name=_attack_name(str(cell.get("attack", "demo_fixture"))),
                severity=int(cell.get("severity", 3)),
                clean_map=float(report.get("ap_clean", 0.0)),
                attacked_map=float(cell.get("ap", 0.0)),
                map_drop_pct=degradation,
                clean_conf=clean_conf,
                attacked_conf=attacked_conf,
                psnr="28.40 dB",
                ssim="0.884",
                inference_ms=48.2,
                robustness_score=max(0.0, 100.0 - degradation),
                clean_bbox_count=len(clean_items),
                attacked_bbox_count=len(attacked_items),
                sample_id=str(sample.get("sample_id", "demo/scene-01")),
                clean_miou=float(report.get("metrics", {}).get("clean", {}).get("miou", 0.0)) or None,
                attacked_miou=float(cell.get("metrics", {}).get("miou", 0.0)) or None,
                miou_drop_pct=degradation if fixture.task_id == "segmentation" else None,
                is_combined=False,
                attack_components_json=json.dumps([str(cell.get("attack", "demo_fixture"))]),
                note="Prepared demo fixture — no model inference was executed.",
                seed=20260906,
                run_config_hash=f"demo-{fixture.fixture_id}",
                backend_run_id=fixture.baseline_run_id,
            )
        )


def _mean_score(items: list[dict[str, Any]]) -> float:
    values = [float(item.get("score", 0.0)) for item in items if item.get("score") is not None]
    return round(sum(values) / len(values), 3) if values else 0.0


def _class_mapping(fixture: FakeFixture) -> dict[str, str]:
    if fixture.task_id == "segmentation":
        return {"Road": "Road", "Car": "Car", "Building": "Building"}
    return {"Car": "Car"}


def _project_name(fixture: FakeFixture) -> str:
    names = {
        "demo-robust-001": "Demo 01 · Vehicle Safety Robustness",
        "demo-seg-002": "Demo 02 · Urban Scene Segmentation",
        "demo-review-003": "Demo 03 · Production Review Drill",
    }
    return names[fixture.fixture_id]


def _project_description(fixture: FakeFixture) -> str:
    return f"DEMO / SIMULATED — prepared {fixture.task_name} walkthrough."


def _session_name(fixture: FakeFixture) -> str:
    return {
        "demo-robust-001": "Vehicle Safety Robustness — Attack → Review → Defence",
        "demo-seg-002": "Urban Scene Segmentation — Mask Recovery Drill",
        "demo-review-003": "Production Review Drill — Block Deploy → Recovery",
    }[fixture.fixture_id]


def _session_description(fixture: FakeFixture) -> str:
    return f"DEMO / SIMULATED — {fixture.dataset_name} with precomputed evidence."


def _attack_name(attack: str) -> str:
    return {
        "occlusion": "Occlusion Patch",
        "motion_blur": "Motion Blur",
        "brightness": "Brightness Shift",
    }.get(attack, attack.replace("_", " ").title())
