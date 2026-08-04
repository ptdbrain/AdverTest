from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipeline.benchmark import (
    AttackBenchmarkConfig,
    AttackDatasetBenchmark,
    BenchmarkModelConfig,
)
from src.pipeline.generator import AttackDatasetGenerator, AttackGenerationConfig


def _generation(tmp_path: Path) -> Path:
    report = AttackDatasetGenerator().generate(
        AttackGenerationConfig(
            dataset_name="synthetic_shapes",
            dataset_params={"n_samples": 2, "seed": 41},
            attack_name="gaussian_noise",
            attack_params={"sigma_per_severity": [0.01, 0.03, 0.06]},
            severities=[1, 3],
            seed=17,
            output_dir=str(tmp_path / "attacked"),
            preview=False,
            limit=2,
        )
    )
    return report.root


def test_benchmark_pairs_clean_and_generated_samples(tmp_path: Path) -> None:
    generation = _generation(tmp_path)
    artifacts = AttackDatasetBenchmark().run(
        AttackBenchmarkConfig(
            generation_paths=[str(generation)],
            model=BenchmarkModelConfig(name="blob_detector"),
            output_dir=str(tmp_path / "benchmarks"),
        )
    )
    report = json.loads(artifacts.report_path.read_text(encoding="utf-8"))

    assert artifacts.summary_path.is_file()
    assert report["format"] == "advertest-benchmark-v1"
    assert report["score_threshold"] == pytest.approx(0.001)
    assert report["n_cells"] == 2
    assert [cell["severity"] for cell in report["cells"]] == [1, 3]
    assert all(cell["n_samples"] == 2 for cell in report["cells"])
    assert all("attack_success" in cell for cell in report["cells"])
    assert all(cell["white_box_same_checkpoint"] is False for cell in report["cells"])
    summary = artifacts.summary_path.read_text(encoding="utf-8")
    assert "attack_success_rate" in summary.splitlines()[0]


def test_benchmark_rejects_changed_clean_source(tmp_path: Path) -> None:
    generation = _generation(tmp_path)
    config_path = generation / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["dataset_params"]["seed"] = 999
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="clean source hash changed"):
        AttackDatasetBenchmark().run(
            AttackBenchmarkConfig(
                generation_paths=[str(generation)],
                model=BenchmarkModelConfig(name="blob_detector"),
                output_dir=str(tmp_path / "benchmarks"),
            )
        )


def test_yolo_benchmark_requires_explicit_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit checkpoint"):
        AttackDatasetBenchmark().run(
            AttackBenchmarkConfig(
                generation_paths=[str(tmp_path / "missing")],
                model=BenchmarkModelConfig(name="yolo11"),
                output_dir=str(tmp_path / "benchmarks"),
            )
        )
