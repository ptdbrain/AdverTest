from __future__ import annotations

import asyncio

import src.main as main_module


class _ProductionSettings:
    app_env = "production"
    bootstrap_demo_model = False
    bootstrap_demo_kitti = True
    bootstrap_drive_export_catalog = False
    bootstrap_demo_catalog = False
    checkpoint_root = "/tmp/checkpoints"
    bootstrap_demo_model_id = "yolo11n"
    demo_model_storage_key = "unused"
    data_root = "/tmp/data"
    drive_export_kitti2d_storage_prefix = "catalog/datasets/kitti2d-100/v1/"
    drive_export_cityscapes_storage_prefix = "catalog/datasets/cityscapes-instance-100/v1/"
    drive_export_nuscenes_storage_prefix = "catalog/datasets/nuscenes-mini-100/v1/"
    app_name = "AdverTest"
    allow_dev_bootstrap_accounts = False

    def validate_production_environment(self) -> None:
        pass


def test_production_startup_skips_retired_legacy_kitti_bootstrap(monkeypatch) -> None:
    """A removed optional demo prefix must not make the Render API unavailable."""
    legacy_kitti_calls: list[object] = []
    drive_catalog_calls: list[object] = []
    storage = object()

    monkeypatch.setattr(main_module, "get_settings", lambda: _ProductionSettings())
    monkeypatch.setattr(main_module, "get_platform_storage", lambda: storage)
    monkeypatch.setattr(main_module, "ensure_demo_kitti", lambda **kwargs: legacy_kitti_calls.append(kwargs))
    monkeypatch.setattr(main_module, "ensure_drive_export_catalog", lambda **kwargs: drive_catalog_calls.append(kwargs))
    monkeypatch.setattr(main_module, "load_attacks", lambda: [])
    monkeypatch.setattr(main_module, "load_adapters", lambda: [])
    monkeypatch.setattr(main_module, "load_datasets", lambda: [])

    async def start_and_stop() -> None:
        async with main_module.lifespan(main_module.app):
            pass

    asyncio.run(start_and_stop())

    assert legacy_kitti_calls == []
    assert drive_catalog_calls == []
