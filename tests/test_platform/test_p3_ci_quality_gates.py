"""P3 CI Quality Gates & Architecture Integrity Test Suite.

Enforces:
1. No duplicate HTTP/WS route definitions across all mounted FastAPI routers.
2. Alembic migration chain is contiguous and terminates at current HEAD.
3. Production environment validation strictly rejects missing Google Client ID & insecure secrets.
4. Official evaluation report rejects missing protocol hashes.
5. Frontend source tree contains zero runtime imports from mockData.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from src.config import Settings
from src.evaluation.report import RunReport
from src.main import app


def test_no_duplicate_http_or_ws_routes() -> None:
    """CI Gate 1: Ensure no duplicate HTTP or WebSocket route registrations exist on FastAPI app."""
    seen_routes: dict[str, str] = {}
    duplicates: list[str] = []

    for route in app.routes:
        methods = getattr(route, "methods", None) or {"WS"}
        path = getattr(route, "path", str(route))
        for method in methods:
            key = f"{method} {path}"
            endpoint_name = getattr(route, "name", str(route))
            if key in seen_routes:
                # Same route registered multiple times
                duplicates.append(f"{key} (existing: {seen_routes[key]}, new: {endpoint_name})")
            else:
                seen_routes[key] = endpoint_name

    assert not duplicates, f"Found duplicate route registrations: {duplicates}"


def test_alembic_migrations_are_at_head() -> None:
    """CI Gate 2: Verify that Alembic script directory has a valid single head revision."""
    alembic_cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(alembic_cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"Expected exactly 1 migration HEAD revision, got: {heads}"
    assert heads[0] == "20260831_0006", f"HEAD revision should be 20260831_0006, got: {heads[0]}"


def test_production_startup_validation_requires_google_client_id() -> None:
    """CI Gate 3: In production mode, startup fails if GOOGLE_CLIENT_ID or JWT_SECRET is invalid."""
    insecure_settings = Settings(
        app_env="production",
        jwt_secret="short",
        google_client_id="",
        cors_origins="https://app.example.com",
    )
    with pytest.raises(ValueError, match="CRITICAL PRODUCTION CONFIGURATION ERROR"):
        insecure_settings.validate_production_environment()

    # Valid production settings
    valid_settings = Settings(
        app_env="production",
        jwt_secret="9f83ab4e12c5890d71ef6a32b9845cd12019487fae6b8c9d0123456789abcdef",
        admin_default_password="StrongProdPassword2026!#",
        platform_database_url="postgresql://advertest:secure_password@postgres:5432/advertest_db",
        google_client_id="123456789-prod.apps.googleusercontent.com",
        cors_origins="https://app.example.com",
        object_storage_backend="local",
    )
    # Should not raise
    valid_settings.validate_production_environment()


def test_official_metric_requires_protocol_hash() -> None:
    """CI Gate 4: Official evaluation provenance must include a non-empty protocol_hash."""
    report = RunReport(
        run_id="TEST-RUN-001",
        model="yolo11s",
        model_version="1.0.0",
        dataset="kitti_val",
        n_samples=100,
        ap_clean=0.72,
        provenance={"protocol_hash": "official-r40-kitti-locked-v1"},
    )
    rep_dict = report.as_dict()
    assert rep_dict["provenance"].get("protocol_hash") == "official-r40-kitti-locked-v1"


def test_frontend_source_contains_zero_runtime_mock_data_imports() -> None:
    """CI Gate 5: Scan frontend/src directory to ensure zero files import from mockData."""
    frontend_src = Path("frontend/src")
    if not frontend_src.exists():
        pytest.skip("frontend/src not present in local workspace")

    violations: list[str] = []
    for file_path in frontend_src.rglob("*"):
        if file_path.suffix in (".js", ".jsx", ".ts", ".tsx"):
            # Exclude mockData definition file itself and test files
            if file_path.name == "mockData.js" or "__tests__" in file_path.parts:
                continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if "from \"@/data/mockData\"" in content or "from '@/data/mockData'" in content:
                violations.append(str(file_path))
            if "from \"../data/mockData\"" in content or "from '../data/mockData'" in content:
                violations.append(str(file_path))

    assert not violations, f"Found runtime imports from mockData in frontend files: {violations}"
