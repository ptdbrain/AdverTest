from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.types import Sample
from src.datasets.base import DatasetSource
from src.datasets.contracts import SplitManifest
from src.datasets.splits import SplitBuilder, SplitPolicy
from src.datasets.versioning import DatasetIngestor, IngestConfig


class SplitSource(DatasetSource):
    name = "split-fixture"
    anonymized = True
    loader_version = "split-loader-1"

    def __init__(self, count: int, *, official: bool = False) -> None:
        self._count = count
        self._official = official
        super().__init__()

    def load(self, limit: int | None = None) -> list[Sample]:
        samples = []
        labels = ("Car", "Pedestrian", "Cyclist")
        for index in range(self._count):
            split = ("train", "val", "test")[index % 3] if self._official else None
            samples.append(
                Sample(
                    sample_id=f"sample-{index:03d}",
                    image=np.full((2, 2, 3), index / 100, dtype=np.float32),
                    anonymized=True,
                    meta={
                        "source_uri": f"split-fixture://sample-{index:03d}",
                        "loader_version": self.loader_version,
                        "native_labels": (labels[index % len(labels)],),
                        "class_labels": (labels[index % len(labels)],),
                        "split": split,
                    },
                )
            )
        return samples[:limit]


def _version(tmp_path: Path, *, count: int = 20, official: bool = False):
    return DatasetIngestor(tmp_path / "versions").ingest(
        SplitSource(count, official=official),
        IngestConfig(name="split-fixture", logical_source_id="split-fixture"),
    )


def test_seeded_split_is_order_independent_and_70_15_15(tmp_path: Path) -> None:
    version = _version(tmp_path)
    builder = SplitBuilder()
    policy = SplitPolicy(strategy="seeded", seed=195)

    first = builder.build(version, policy)
    reordered = version.model_copy(update={"records": tuple(reversed(version.records))})
    second = builder.build(reordered, policy)

    assert first.train_ids == second.train_ids
    assert first.val_ids == second.val_ids
    assert first.test_ids == second.test_ids
    assert (len(first.train_ids), len(first.val_ids), len(first.test_ids)) == (14, 3, 3)
    assert builder.validate(version, first).passed


def test_official_split_membership_is_preserved_and_test_is_locked(tmp_path: Path) -> None:
    version = _version(tmp_path, count=12, official=True)
    manifest = SplitBuilder().build(
        version,
        SplitPolicy(strategy="official", seed=195, lock_test=True),
    )

    expected_test = tuple(
        record.sample_id for record in version.records if record.official_split == "test"
    )
    assert manifest.test_ids == expected_test
    assert manifest.locked_test_ids == expected_test

    tampered = manifest.model_copy(
        update={
            "train_ids": (*manifest.train_ids, manifest.test_ids[0]),
            "test_ids": manifest.test_ids[1:],
        }
    )
    report = SplitBuilder().validate(version, tampered)
    assert not report.passed
    assert "locked_test_membership_changed" in report.errors


def test_class_stratified_split_has_no_overlap(tmp_path: Path) -> None:
    version = _version(tmp_path, count=30)
    manifest = SplitBuilder().build(
        version,
        SplitPolicy(strategy="class_stratified", seed=7),
    )

    train = set(manifest.train_ids)
    val = set(manifest.val_ids)
    test = set(manifest.test_ids)
    assert not (train & val or train & test or val & test)
    assert train | val | test == {record.sample_id for record in version.records}
    for split_ids in (train, val, test):
        present = {
            label
            for record in version.records
            if record.sample_id in split_ids
            for label in record.class_labels
        }
        assert present == {"Car", "Pedestrian", "Cyclist"}


def test_split_manifest_is_immutable(tmp_path: Path) -> None:
    manifest: SplitManifest = SplitBuilder().build(
        _version(tmp_path),
        SplitPolicy(strategy="seeded", seed=1),
    )

    try:
        manifest.test_ids = ()  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("SplitManifest must be frozen")
