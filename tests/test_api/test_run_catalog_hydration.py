from __future__ import annotations

from fastapi import HTTPException

from src.api.routers import runs
from src.pipeline.runner import RunConfig


class _Settings:
    def __init__(self, data_root: str) -> None:
        self.data_root = data_root
        self.drive_export_kitti2d_storage_prefix = "catalog/datasets/kitti2d-100/v1/"


def test_preflight_hydration_fetches_only_the_catalog_owned_kitti_root(monkeypatch, tmp_path) -> None:
    settings = _Settings(str(tmp_path / "data"))
    calls: list[dict] = []
    expected_root = tmp_path / "data" / "catalog" / "kitti2d-100"
    monkeypatch.setattr(runs, "get_settings", lambda: settings)
    monkeypatch.setattr(runs, "ensure_drive_export_bundle", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr("src.api.platform_dependencies.get_platform_storage", lambda: "storage")

    runs._hydrate_selected_catalog_bundle(
        RunConfig(dataset="kitti", dataset_params={"root": str(expected_root)})
    )

    assert calls == [
        {
            "storage": "storage",
            "storage_prefix": "catalog/datasets/kitti2d-100/v1/",
            "data_root": str(tmp_path / "data"),
            "bundle_name": "kitti2d-100",
        }
    ]


def test_preflight_hydration_never_fetches_for_a_user_supplied_root(monkeypatch, tmp_path) -> None:
    settings = _Settings(str(tmp_path / "data"))
    monkeypatch.setattr(runs, "get_settings", lambda: settings)
    monkeypatch.setattr(runs, "ensure_drive_export_bundle", lambda **kwargs: (_ for _ in ()).throw(AssertionError))

    runs._hydrate_selected_catalog_bundle(
        RunConfig(dataset="kitti", dataset_params={"root": str(tmp_path / "user-upload")})
    )


def test_platform_run_scope_uses_already_checked_actor(monkeypatch) -> None:
    monkeypatch.setattr(runs, "_remote_enabled", lambda: True)
    assert runs._project_scope("project-member", "user-member") == "project-member"


def test_platform_run_scope_rejects_missing_actor(monkeypatch) -> None:
    monkeypatch.setattr(runs, "_remote_enabled", lambda: True)
    try:
        runs._project_scope("project-member", None)
    except HTTPException as exc:
        assert exc.status_code == 401
    else:  # pragma: no cover - assertion guard
        raise AssertionError("expected authenticated actor requirement")
