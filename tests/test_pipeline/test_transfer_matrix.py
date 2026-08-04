from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.benchmark import (
    BenchmarkModelConfig,
    TransferMatrixBenchmark,
    TransferMatrixConfig,
)
from src.pipeline.generator import AttackDatasetGenerator, AttackGenerationConfig, SurrogateConfig


def test_transfer_matrix_benchmarks_existing_generation(tmp_path: Path) -> None:
    generation = AttackDatasetGenerator().generate(
        AttackGenerationConfig(
            dataset_name="synthetic_shapes",
            dataset_params={"n_samples": 2, "seed": 41},
            attack_name="pgd",
            attack_params={"steps": 1, "restarts": 1},
            severities=[1],
            surrogate=SurrogateConfig(name="blob_detector"),
            output_dir=str(tmp_path / "attacked"),
            preview=False,
        )
    )
    artifacts = TransferMatrixBenchmark().run(
        TransferMatrixConfig(
            generation_paths=[str(generation.root)],
            target_models=[BenchmarkModelConfig(name="blob_detector")],
            output_dir=str(tmp_path / "transfer"),
        )
    )
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))
    assert report["format"] == "advertest-transfer-matrix-v1"
    assert report["rows"][0]["source"]["surrogate"] == "blob_detector"
    assert report["rows"][0]["targets"]["blob_detector"]["n_cells"] == 1
