"""KITTI loader: label parsing, difficulty filter, and the anonymisation gate.

The fixture builds a miniature KITTI tree in ``tmp_path``, so nothing here needs
the 12 GB download. Tests that decode pixels need Pillow (a local extra, kept out
of ``requirements.txt``) and skip without it; label parsing and the gate do not.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.core.types import Box
from src.datasets import get_dataset
from src.datasets.base import AnonymizationRequiredError
from src.datasets.kitti import Kitti
from src.datasets.kitti_anonymize import anonymize_placeholder

#: Pillow is a local extra (never in CI), so pixel tests opt out without it.
HAS_PILLOW = importlib.util.find_spec("PIL") is not None
needs_pillow = pytest.mark.skipif(not HAS_PILLOW, reason="KITTI pixel tests need pillow")

IMAGE_HEIGHT, IMAGE_WIDTH = 120, 200

#: type trunc occl alpha x1 y1 x2 y2 h w l x y z ry
LABEL_ROWS = {
    "000000": [
        "Car 0.00 0 -1.57 20.00 30.00 90.00 100.00 1.5 1.6 4.0 1 2 3 -1.5",
        "Pedestrian 0.00 0 -1.57 120.00 20.00 140.00 90.00 1.8 0.6 0.8 1 2 3 -1.5",
        "DontCare 0.00 0 -10 0.00 0.00 10.00 10.00 -1 -1 -1 -1000 -1000 -1000 -10",
        "Van 0.00 0 -1.57 150.00 10.00 190.00 60.00 2.0 1.8 5.0 1 2 3 -1.5",
    ],
    "000001": [
        # 10 px tall: dropped by every difficulty band except "all".
        "Car 0.00 0 -1.57 10.00 10.00 60.00 20.00 1.5 1.6 4.0 1 2 3 -1.5",
        # Heavily occluded and truncated: survives "hard" and "all" only.
        "Cyclist 0.40 2 -1.57 30.00 30.00 70.00 110.00 1.7 0.6 1.8 1 2 3 -1.5",
    ],
    "000002": ["Car 0.00 0 -1.57 5.00 5.00 95.00 105.00 1.5 1.6 4.0 1 2 3 -1.5"],
}


def _write_png(path: Path, seed: int) -> None:
    from PIL import Image

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


@pytest.fixture
def kitti_root(tmp_path: Path) -> Path:
    """Miniature KITTI tree: 3 frames, labels always, PNGs when pillow is around."""
    root = tmp_path / "kitti"
    images = root / "training" / "image_2"
    labels = root / "training" / "label_2"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index, (image_id, rows) in enumerate(LABEL_ROWS.items()):
        (labels / f"{image_id}.txt").write_text("\n".join(rows) + "\n")
        if HAS_PILLOW:
            _write_png(images / f"{image_id}.png", seed=index)
    (root / "ImageSets").mkdir()
    (root / "ImageSets" / "val.txt").write_text("000000\n000001\n000002\n")
    (root / "ImageSets" / "train.txt").write_text("000000\n")
    return root


def _dataset(root: Path, **params: object) -> Kitti:
    return get_dataset("kitti", root=str(root), manifest_path=str(root / "manifest.json"), **params)  # type: ignore[return-value]


# ----------------------------------------------------------------- label parsing


def test_labels_map_onto_the_three_normalised_classes(kitti_root: Path) -> None:
    dataset = _dataset(kitti_root, difficulty="moderate")
    boxes, dropped = dataset._read_labels(
        kitti_root / "training" / "label_2" / "000000.txt", (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    assert [box.label for box in boxes] == ["Car", "Pedestrian"]
    assert dropped["DontCare"] == 1
    assert dropped["Van"] == 1


def test_merge_van_truck_folds_vans_into_car(kitti_root: Path) -> None:
    dataset = _dataset(kitti_root, merge_van_truck=True)
    boxes, dropped = dataset._read_labels(
        kitti_root / "training" / "label_2" / "000000.txt", (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    assert [box.label for box in boxes] == ["Car", "Pedestrian", "Car"]
    assert "Van" not in dropped


@pytest.mark.parametrize(
    ("difficulty", "expected"),
    [("easy", 0), ("moderate", 0), ("hard", 1), ("all", 2)],
)
def test_difficulty_bands_filter_small_and_occluded_boxes(
    kitti_root: Path, difficulty: str, expected: int
) -> None:
    dataset = _dataset(kitti_root, difficulty=difficulty)
    boxes, _ = dataset._read_labels(
        kitti_root / "training" / "label_2" / "000001.txt", (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    assert len(boxes) == expected


def test_boxes_are_clipped_to_the_frame(kitti_root: Path) -> None:
    dataset = _dataset(kitti_root, difficulty="all")
    boxes, _ = dataset._read_labels(
        kitti_root / "training" / "label_2" / "000002.txt", (IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    box = boxes[0]
    assert 0.0 <= box.x1 < box.x2 <= IMAGE_WIDTH
    assert 0.0 <= box.y1 < box.y2 <= IMAGE_HEIGHT


# --------------------------------------------------------------- the §6 gate


def test_anonymize_off_keeps_the_gate_closed(kitti_root: Path) -> None:
    """No bypass flag exists: an un-anonymised KITTI must not reach a model."""
    dataset = _dataset(kitti_root, anonymize="off")
    assert dataset.anonymized is False
    with pytest.raises(AnonymizationRequiredError):
        dataset.require_anonymized()


def test_placeholder_opens_the_gate(kitti_root: Path) -> None:
    dataset = _dataset(kitti_root, anonymize="placeholder")
    assert dataset.anonymized is True
    dataset.require_anonymized()
    assert "PLACEHOLDER" in dataset.info().note


def test_catalog_entry_reports_kitti_as_not_anonymised() -> None:
    """The class-level catalog must not claim KITTI ships anonymised."""
    assert Kitti.describe()["anonymized"] is False


# --------------------------------------------------------------------- loading


@needs_pillow
def test_load_returns_samples_with_ids_and_metadata(kitti_root: Path) -> None:
    samples = _dataset(kitti_root, difficulty="all").load()
    assert len(samples) == 3
    first = samples[0]
    assert first.meta["image_id"] == "000000"
    assert first.image.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    assert first.image.dtype == np.float32
    assert 0.0 <= float(first.image.min()) and float(first.image.max()) <= 1.0
    assert first.anonymized is True


@needs_pillow
def test_sample_ids_select_exactly_those_frames(kitti_root: Path) -> None:
    samples = _dataset(kitti_root, sample_ids=("000002",)).load()
    assert [sample.meta["image_id"] for sample in samples] == ["000002"]


@needs_pillow
def test_sample_id_changes_when_the_content_changes(kitti_root: Path) -> None:
    """Anonymisation and difficulty alter pixels/GT, so they belong in the cache key."""
    anonymised = _dataset(kitti_root, anonymize="placeholder").load(1)[0]
    raw = _dataset(kitti_root, anonymize="off").load(1)[0]
    harder = _dataset(kitti_root, difficulty="all").load(1)[0]
    assert len({anonymised.sample_id, raw.sample_id, harder.sample_id}) == 3


@needs_pillow
def test_split_file_is_honoured(kitti_root: Path) -> None:
    samples = _dataset(kitti_root, split="train").load()
    assert [sample.meta["image_id"] for sample in samples] == ["000000"]


@needs_pillow
def test_limit_is_applied_before_decoding(kitti_root: Path) -> None:
    assert len(_dataset(kitti_root).load(2)) == 2


@needs_pillow
def test_manifest_is_written_for_every_loaded_frame(kitti_root: Path) -> None:
    _dataset(kitti_root).load()
    manifest = json.loads((kitti_root / "manifest.json").read_text())
    assert manifest["n_frames"] == 3
    assert "PLACEHOLDER" in manifest["warning"]
    assert all(entry["digest"] for entry in manifest["entries"])


def test_missing_root_names_the_download_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_kitti.sh"):
        get_dataset("kitti", root=str(tmp_path / "nope")).load()


# ------------------------------------------------------------- the anonymiser


def test_placeholder_blurs_only_inside_the_derived_regions() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((60, 80, 3), dtype=np.float32)
    boxes = (Box(10.0, 10.0, 40.0, 50.0, "Car"),)
    result = anonymize_placeholder(image, boxes)
    assert result.n_regions == 1
    changed = np.any(result.image != image, axis=2)
    # Plate band: bottom 40 % of the box, nothing outside it.
    assert changed[10:30, 10:40].sum() == 0
    assert changed[35:50, 10:40].any()
    assert changed[:, 45:].sum() == 0


def test_person_boxes_are_blurred_at_the_head_not_the_feet() -> None:
    rng = np.random.default_rng(1)
    image = rng.random((60, 80, 3), dtype=np.float32)
    result = anonymize_placeholder(image, (Box(10.0, 10.0, 30.0, 50.0, "Pedestrian"),))
    changed = np.any(result.image != image, axis=2)
    assert changed[10:22, 10:30].any()
    assert changed[30:50, 10:30].sum() == 0


def test_frame_without_boxes_is_left_alone() -> None:
    image = np.full((20, 20, 3), 0.4, dtype=np.float32)
    result = anonymize_placeholder(image, ())
    assert result.n_regions == 0
    assert np.array_equal(result.image, image)
