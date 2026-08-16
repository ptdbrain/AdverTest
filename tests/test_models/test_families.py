import pytest

from src.config import Settings
from src.models.families import adapter_request
from src.models.versions import ModelVersion


def _version(model_name: str, task: str, metadata: dict | None = None) -> ModelVersion:
    return ModelVersion("model", model_name, task, "checkpoint.pt", "hash", None, metadata or {}, True)


def test_yolo_family_owns_yolo_specific_constructor_arguments() -> None:
    adapter, params = adapter_request(_version("yolo11s", "detection2d"), checkpoint="checkpoint.pt", config=type("Run", (), {"confidence_threshold": 0.61})(), settings=Settings())

    assert adapter == "yolo11"
    assert params["score_threshold"] == 0.61
    assert "batch_size" in params


def test_sam_family_never_receives_yolo_constructor_arguments() -> None:
    adapter, params = adapter_request(_version("sam2", "segmentation", {"sam_config": "sam2.yaml"}), checkpoint="checkpoint.pt", config=type("Run", (), {"confidence_threshold": 0.5})(), settings=Settings())

    assert adapter == "sam2"
    assert params == {"weights": "checkpoint.pt", "config": "sam2.yaml", "device": "cpu", "mask_threshold": 0.5}


def test_sam_family_requires_its_config() -> None:
    with pytest.raises(ValueError, match="MODEL_FAMILY_CONFIG_MISSING"):
        adapter_request(_version("sam2", "segmentation"), checkpoint="checkpoint.pt", config=type("Run", (), {"confidence_threshold": 0.5})(), settings=Settings())
