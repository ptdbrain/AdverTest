from __future__ import annotations

import pytest

from src.models.catalog import ModelVersionCatalog, ModelVersionUnavailableError
from src.models.versions import ModelVersion


def _version(*, runnable: bool) -> ModelVersion:
    return ModelVersion(
        id="yolo11s-kitti-clean-b0",
        model_name="yolo11s",
        task="detection2d",
        checkpoint_path="C:/models/best.pt" if runnable else None,
        checkpoint_hash="abc" if runnable else None,
        parent_id=None,
        training_metadata={},
        runnable=runnable,
        blocked_reason=None if runnable else "CHECKPOINT_MISSING",
    )


def test_catalog_filters_versions_by_task() -> None:
    catalog = ModelVersionCatalog([_version(runnable=True)])

    assert catalog.list(task="detection2d") == [_version(runnable=True)]
    assert catalog.list(task="segmentation") == []


def test_catalog_refuses_a_visible_but_non_runnable_version() -> None:
    catalog = ModelVersionCatalog([_version(runnable=False)])

    with pytest.raises(ModelVersionUnavailableError, match="CHECKPOINT_MISSING"):
        catalog.require_runnable("yolo11s-kitti-clean-b0")
