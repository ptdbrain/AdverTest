from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.attacks import get_attack
from src.attacks.base import AttackContext
from src.core.types import Box, Box3D, CameraView, LidarFrame, Sample
from src.datasets import get_dataset
from src.pipeline.generator import AttackDatasetGenerator, AttackGenerationConfig, inspect_generated_dataset


class SensorFixtureSource:
    name = "sensor_fixture"

    def __init__(self, sample: Sample) -> None:
        self.sample = sample

    def require_anonymized(self) -> None:
        assert self.sample.anonymized

    def load(self, limit: int | None = None) -> list[Sample]:
        return [self.sample] if limit is None or limit else []


def _sample() -> Sample:
    image = np.full((12, 16, 3), 0.5, dtype=np.float32)
    depth = np.full((12, 16), 20.0, dtype=np.float32)
    cameras = (
        CameraView(
            "CAM_FRONT",
            image.copy(),
            depth=depth.copy(),
            intrinsic=np.eye(3, dtype=np.float32),
            sensor_to_ego=np.eye(4, dtype=np.float32),
            previous_image=np.full_like(image, 0.25),
        ),
        CameraView(
            "CAM_BACK",
            np.full_like(image, 0.6),
            depth=depth.copy(),
            intrinsic=np.eye(3, dtype=np.float32),
            sensor_to_ego=np.eye(4, dtype=np.float32),
        ),
    )
    return Sample(
        sample_id="sensor_001",
        image=image,
        boxes=(Box(1, 1, 8, 9, "Car"),),
        boxes3d=(Box3D(1, 2, 3, 4, 2, 1.5, 0.1, "Car"),),
        camera_views=cameras,
        lidar_frame=LidarFrame(
            np.array([[1, 2, 3, 0.7, 4]], dtype=np.float32),
            sensor_model="fixture-lidar",
        ),
        anonymized=True,
    )


def test_depth_weather_requires_depth_unless_prior_is_explicit() -> None:
    sample = _sample().with_image(np.full((12, 16, 3), 0.5, dtype=np.float32))
    sample = Sample(sample.sample_id, sample.image, boxes=sample.boxes, anonymized=True)
    with pytest.raises(ValueError, match="depth-aware weather requires"):
        get_attack("depth_fog", depth_policy="required").run(
            sample,
            1,
            AttackContext(np.random.default_rng(1)),
        )
    attacked = get_attack("depth_fog").run(
        sample,
        1,
        AttackContext(np.random.default_rng(1)),
    )
    assert not np.array_equal(attacked.image, sample.image)


def test_multimodal_generation_round_trip_and_tamper_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = SensorFixtureSource(_sample())
    monkeypatch.setattr(AttackDatasetGenerator, "_source", staticmethod(lambda _: source))
    config = AttackGenerationConfig(
        dataset_name="sensor_fixture",
        attack_name="depth_fog",
        severities=[1],
        output_dir=str(tmp_path),
        preview=False,
    )
    report = AttackDatasetGenerator().generate(config)
    loaded = get_dataset("generated_dataset", root=str(report.root)).load()[0]
    assert len(loaded.camera_views) == 2
    assert loaded.camera_views[0].intrinsic is not None
    assert loaded.camera_views[0].previous_image is not None
    assert loaded.lidar_frame is not None
    assert loaded.lidar_frame.sensor_model == "fixture-lidar"
    assert loaded.boxes3d == source.sample.boxes3d
    record = json.loads((report.root / "manifest.jsonl").read_text(encoding="utf-8"))
    camera_path = report.root / record["camera_payloads"][0]["image_path"]
    np.save(camera_path, np.zeros((2, 2, 3), dtype=np.float32), allow_pickle=False)
    inspected = inspect_generated_dataset(report.root)
    assert inspected["valid"] is False
    assert inspected["invalid_variants"] == [record["variant_id"]]
