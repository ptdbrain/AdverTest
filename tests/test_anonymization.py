from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.anonymization import (
    AnonymizationConfig,
    DatasetAnonymizer,
    Detection,
    inspect_anonymized_dataset,
)
from src.anonymization.pipeline import DetectorConfig
from src.core.hashing import file_digest
from src.datasets.io import load_image


class FakeDetector:
    def __init__(
        self,
        kind: str,
        box: tuple[float, float, float, float],
        *,
        checkpoint_hash: str | None = None,
    ) -> None:
        self.kind = kind
        self.box = box
        self.name = f"fake:{kind}"
        self.checkpoint_hash = checkpoint_hash or f"{kind}-hash"

    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        return (
            Detection(
                kind=self.kind,  # type: ignore[arg-type]
                x1=self.box[0],
                y1=self.box[1],
                x2=self.box[2],
                y2=self.box[3],
                score=0.9,
                detector=self.name,
            ),
        )


class FailingDetector(FakeDetector):
    def detect(self, image: np.ndarray) -> tuple[Detection, ...]:
        raise RuntimeError("detector failed")


def _write_kitti(root: Path, count: int = 2) -> None:
    (root / "image_2").mkdir(parents=True)
    (root / "label_2").mkdir()
    for index in range(count):
        horizontal = np.tile(np.arange(32, dtype=np.uint8), (24, 1))
        image = np.stack(
            [
                horizontal * 7,
                np.roll(horizontal, index + 1, axis=1) * 5,
                horizontal * 3,
            ],
            axis=-1,
        )
        Image.fromarray(image).save(root / "image_2" / f"{index:06d}.png")
        (root / "label_2" / f"{index:06d}.txt").write_text(
            "Car 0 0 0 2 3 20 18 1 1 1 1 1 1 1\n",
            encoding="utf-8",
        )


def _config(source: Path, output: Path) -> AnonymizationConfig:
    return AnonymizationConfig(
        input_dir=str(source),
        output_dir=str(output),
        face_detector=DetectorConfig(checkpoint="face.onnx", expansion=0),
        plate_detector=DetectorConfig(checkpoint="plate.onnx", expansion=0),
        method="gaussian_mosaic",
    )


def _detectors(
    *,
    face_hash: str = "face-hash",
) -> dict[str, FakeDetector]:
    return {
        "face": FakeDetector("face", (2, 3, 10, 12), checkpoint_hash=face_hash),
        "license_plate": FakeDetector("license_plate", (18, 14, 29, 21)),
    }


def test_anonymizer_blurs_regions_preserves_labels_and_resumes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_kitti(source)
    source_hash = file_digest(source / "image_2" / "000000.png", length=64)
    anonymizer = DatasetAnonymizer(_detectors())  # type: ignore[arg-type]

    first = anonymizer.anonymize(_config(source, output))
    second = anonymizer.anonymize(_config(source, output))

    assert first.processed_samples == 2
    assert first.resumed_samples == 0
    assert first.face_detections == 2
    assert first.plate_detections == 2
    assert second.resumed_samples == 2
    assert file_digest(source / "image_2" / "000000.png", length=64) == source_hash
    assert (output / "label_2" / "000000.txt").read_bytes() == (
        source / "label_2" / "000000.txt"
    ).read_bytes()
    assert not np.array_equal(
        load_image(output / "image_2" / "000000.png"),
        load_image(source / "image_2" / "000000.png"),
    )
    descriptor = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert descriptor["status"] == "complete"
    assert descriptor["anonymized"] is True
    assert descriptor["review_status"] == "pending_spot_check"
    assert inspect_anonymized_dataset(output)["valid"] is True


def test_changed_detector_hash_invalidates_resume(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_kitti(source, count=1)
    config = _config(source, output)

    DatasetAnonymizer(_detectors(face_hash="v1")).anonymize(config)  # type: ignore[arg-type]
    rerun = DatasetAnonymizer(_detectors(face_hash="v2")).anonymize(config)  # type: ignore[arg-type]

    assert rerun.resumed_samples == 0
    record = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8"))
    assert record["detector_hashes"]["face"] == "v2"


def test_detector_failure_keeps_dataset_non_anonymized(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_kitti(source, count=1)
    detectors = _detectors()
    detectors["face"] = FailingDetector("face", (1, 1, 2, 2))

    with pytest.raises(RuntimeError, match="detector failed"):
        DatasetAnonymizer(detectors).anonymize(_config(source, output))  # type: ignore[arg-type]

    descriptor = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert descriptor["status"] == "incomplete"
    assert descriptor["anonymized"] is False


def test_inspector_detects_tampered_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_kitti(source, count=1)
    DatasetAnonymizer(_detectors()).anonymize(_config(source, output))  # type: ignore[arg-type]
    (output / "image_2" / "000000.png").write_bytes(b"tampered")

    inspected = inspect_anonymized_dataset(output)

    assert inspected["valid"] is False
    assert inspected["invalid_samples"] == ["000000"]


def test_anonymizer_can_select_explicit_sample_ids(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_kitti(source, count=3)
    config = _config(source, output).model_copy(
        update={"sample_ids": ["000002", "000000"]}
    )

    report = DatasetAnonymizer(_detectors()).anonymize(config)  # type: ignore[arg-type]

    assert report.processed_samples == 2
    assert sorted(path.stem for path in (output / "image_2").glob("*.png")) == [
        "000000",
        "000002",
    ]
    assert inspect_anonymized_dataset(output)["valid"] is True
