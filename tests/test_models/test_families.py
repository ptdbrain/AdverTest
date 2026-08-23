import pytest

from src.config import Settings
from src.models.families import FAMILIES, adapter_request, family_for_version
from src.models.versions import ModelVersion


def _version(model_name: str, task: str, metadata: dict | None = None) -> ModelVersion:
    return ModelVersion("model", model_name, task, "checkpoint.pt", "hash", None, metadata or {}, True)


def test_yolo_family_owns_yolo_specific_constructor_arguments() -> None:
    adapter, params = adapter_request(
        _version("yolo11s", "detection2d"),
        checkpoint="checkpoint.pt",
        config=type("Run", (), {"confidence_threshold": 0.61})(),
        settings=Settings(),
    )

    assert adapter == "yolo11"
    assert params["score_threshold"] == 0.61
    assert "batch_size" in params


def test_sam_family_never_receives_yolo_constructor_arguments() -> None:
    adapter, params = adapter_request(
        _version("sam2", "segmentation", {"sam_config": "sam2.yaml"}),
        checkpoint="checkpoint.pt",
        config=type("Run", (), {"confidence_threshold": 0.5})(),
        settings=Settings(),
    )

    assert adapter == "sam2"
    assert params == {"weights": "checkpoint.pt", "config": "sam2.yaml", "device": "cpu", "mask_threshold": 0.5}


def test_sam_family_requires_its_config() -> None:
    with pytest.raises(ValueError, match="MODEL_FAMILY_CONFIG_MISSING"):
        adapter_request(
            _version("sam2", "segmentation"),
            checkpoint="checkpoint.pt",
            config=type("Run", (), {"confidence_threshold": 0.5})(),
            settings=Settings(),
        )


def test_pointpillars_family_requires_detection3d_and_waits_for_gpu_validation() -> None:
    family = family_for_version(_version("pointpillars", "detection3d"))

    assert family is FAMILIES["pointpillars3d"]
    assert family.supported_tasks == frozenset({"detection3d"})
    assert family.checkpoint_extensions == frozenset({".pth"})
    assert family.runnable is False
    assert family.blocked_reason == "WAITING_FOR_GPU_VALIDATION"


def test_pointpillars_rejects_non_detection3d_versions() -> None:
    with pytest.raises(ValueError, match="MODEL_FAMILY_TASK_MISMATCH: pointpillars3d does not support detection2d"):
        family_for_version(_version("pointpillars", "detection2d"))


def test_pointpillars_adapter_request_contains_config_and_checkpoint() -> None:
    adapter, params = adapter_request(
        _version("pointpillars", "detection3d", {"model_config": "pointpillars_kitti.py"}),
        checkpoint="pointpillars.pth",
        config=type("Run", (), {"confidence_threshold": 0.61})(),
        settings=Settings(),
    )

    assert adapter == "pointpillars"
    assert params == {
        "weights": "pointpillars.pth",
        "config": "pointpillars_kitti.py",
        "device": "cpu",
        "score_threshold": 0.61,
    }


def test_pointpillars_rejects_missing_model_config() -> None:
    with pytest.raises(ValueError, match="MODEL_FAMILY_CONFIG_MISSING: PointPillars requires MMDetection3D config"):
        adapter_request(
            _version("pointpillars", "detection3d"),
            checkpoint="pointpillars.pth",
            config=type("Run", (), {"confidence_threshold": 0.5})(),
            settings=Settings(),
        )
