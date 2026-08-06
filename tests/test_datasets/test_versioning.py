from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.types import Box, Sample
from src.datasets.base import DatasetSource
from src.datasets.versioning import DatasetIngestor, IngestConfig


class StaticSource(DatasetSource):
    name = "static-fixture"
    anonymized = True
    loader_version = "fixture-loader-1"

    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples
        super().__init__()

    def load(self, limit: int | None = None) -> list[Sample]:
        return self._samples[:limit]


def _sample(sample_id: str, value: float, source_path: Path) -> Sample:
    return Sample(
        sample_id=sample_id,
        image=np.full((4, 5, 3), value, dtype=np.float32),
        boxes=(Box(0, 0, 2, 3, "Car"),),
        anonymized=True,
        meta={
            "source_path": str(source_path),
            "source_uri": f"fixture://{sample_id}",
            "loader_version": "fixture-loader-1",
            "native_labels": ("vehicle",),
        },
    )


def test_same_source_produces_same_dataset_version(tmp_path: Path) -> None:
    samples = [
        _sample("sample-b", 0.2, tmp_path / "first" / "b.png"),
        _sample("sample-a", 0.1, tmp_path / "first" / "a.png"),
    ]
    ingestor = DatasetIngestor(tmp_path / "versions")
    config = IngestConfig(name="fixture", logical_source_id="fixture-source")

    first = ingestor.ingest(StaticSource(samples), config)
    second = ingestor.ingest(StaticSource(list(reversed(samples))), config)

    assert first.version_id == second.version_id
    assert first.manifest_hash == second.manifest_hash
    assert [record.sample_id for record in first.records] == ["sample-a", "sample-b"]
    assert (tmp_path / "versions" / first.version_id / "manifest.json").is_file()


def test_dataset_identity_excludes_absolute_source_paths(tmp_path: Path) -> None:
    config = IngestConfig(name="fixture", logical_source_id="fixture-source")
    first = DatasetIngestor(tmp_path / "one").ingest(
        StaticSource([_sample("sample-a", 0.1, tmp_path / "mount-a" / "a.png")]),
        config,
    )
    second = DatasetIngestor(tmp_path / "two").ingest(
        StaticSource([_sample("sample-a", 0.1, tmp_path / "mount-b" / "a.png")]),
        config,
    )

    assert first.version_id == second.version_id
    assert first.manifest_hash == second.manifest_hash
    assert "mount-a" not in first.model_dump_json()
    assert "mount-b" not in second.model_dump_json()


def test_changed_content_changes_dataset_version(tmp_path: Path) -> None:
    config = IngestConfig(name="fixture", logical_source_id="fixture-source")
    ingestor = DatasetIngestor(tmp_path / "versions")

    first = ingestor.ingest(
        StaticSource([_sample("sample-a", 0.1, tmp_path / "a.png")]),
        config,
    )
    changed = ingestor.ingest(
        StaticSource([_sample("sample-a", 0.9, tmp_path / "a.png")]),
        config,
    )

    assert first.version_id != changed.version_id
    assert first.records[0].source_hash != changed.records[0].source_hash
