from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.datasets import get_dataset
from src.pipeline.generator import (
    AttackDatasetGenerator,
    AttackGenerationConfig,
    SurrogateConfig,
    inspect_generated_dataset,
)


def _pgd_config(tmp_path: Path) -> AttackGenerationConfig:
    return AttackGenerationConfig(
        dataset_name="synthetic_shapes",
        dataset_params={"n_samples": 2, "seed": 123},
        attack_name="pgd",
        attack_params={"steps": 2, "restarts": 1},
        severities=[1, 3],
        seed=77,
        surrogate=SurrogateConfig(name="blob_detector"),
        output_dir=str(tmp_path),
        preview=False,
    )


def test_generate_pgd_dataset_round_trips_and_resumes(tmp_path: Path) -> None:
    config = _pgd_config(tmp_path)
    first = AttackDatasetGenerator().generate(config)
    assert first.n_variants == 4
    assert first.resumed_variants == 0
    assert first.estimated_canonical_bytes > 0
    assert first.estimated_gradient_steps == 8

    loaded = get_dataset("generated_dataset", root=str(first.root)).load()
    assert len(loaded) == 4
    assert all(sample.image.dtype == np.float32 for sample in loaded)
    assert all(sample.boxes for sample in loaded)
    assert all(sample.mask is not None for sample in loaded)

    second = AttackDatasetGenerator().generate(config)
    assert second.root == first.root
    assert second.n_variants == 4
    assert second.resumed_variants == 4
    inspected = inspect_generated_dataset(first.root)
    assert inspected["valid"] is True
    assert inspected["manifest_records"] == 4
    assert inspected["source_fingerprint"]
    assert inspected["estimate"]["variants"] == 4
    manifest = [
        json.loads(line)
        for line in (first.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["attack_version"] == "1.0.0" for record in manifest)
    assert all(record["source_sample_hash"] for record in manifest)
    assert all(record["label_hash"] for record in manifest)


def test_generator_keeps_source_and_labels_unchanged(tmp_path: Path) -> None:
    source = get_dataset("synthetic_shapes", n_samples=1, seed=999).load()[0]
    before = source.image.copy()
    config = AttackGenerationConfig(
        dataset_name="synthetic_shapes",
        dataset_params={"n_samples": 1, "seed": 999},
        attack_name="dpatch",
        attack_params={"eot": False, "allow_builtin_patch": True},
        severities=[1],
        output_dir=str(tmp_path),
        preview=False,
    )
    report = AttackDatasetGenerator().generate(config)
    generated = get_dataset("generated_dataset", root=str(report.root)).load()[0]
    assert np.array_equal(source.image, before)
    assert generated.boxes == source.boxes
    assert np.array_equal(generated.mask, source.mask)
    assert not np.array_equal(generated.image, source.image)
    record = json.loads(
        (report.root / "manifest.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["patch_artifact_hash"]
    assert (report.root / record["patch_artifact_path"]).is_file()


def test_incomplete_or_tampered_generation_is_rejected(tmp_path: Path) -> None:
    report = AttackDatasetGenerator().generate(_pgd_config(tmp_path))
    manifest = [
        json.loads(line)
        for line in (report.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    image_path = report.root / manifest[0]["image_path"]
    np.save(image_path, np.zeros((2, 2, 3), dtype=np.float32), allow_pickle=False)
    inspected = inspect_generated_dataset(report.root)
    assert inspected["valid"] is False
    assert manifest[0]["variant_id"] in inspected["invalid_variants"]


def test_severity_zero_exports_identity_variant(tmp_path: Path) -> None:
    config = AttackGenerationConfig(
        dataset_name="synthetic_shapes",
        dataset_params={"n_samples": 1, "seed": 12},
        attack_name="pgd",
        attack_params={"steps": 2},
        severities=[0],
        surrogate=SurrogateConfig(name="blob_detector"),
        output_dir=str(tmp_path),
        preview=False,
    )
    report = AttackDatasetGenerator().generate(config)
    source = get_dataset("synthetic_shapes", n_samples=1, seed=12).load()[0]
    generated = get_dataset("generated_dataset", root=str(report.root)).load()[0]
    assert np.array_equal(generated.image, source.image)


def test_tampered_label_invalidates_generation(tmp_path: Path) -> None:
    report = AttackDatasetGenerator().generate(_pgd_config(tmp_path))
    record = json.loads(
        (report.root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    (report.root / record["label_path"]).write_text(
        '{"boxes": []}\n',
        encoding="utf-8",
    )
    inspected = inspect_generated_dataset(report.root)
    assert inspected["valid"] is False
    assert record["variant_id"] in inspected["invalid_variants"]
    with pytest.raises(ValueError, match="label hash mismatch"):
        get_dataset("generated_dataset", root=str(report.root)).load()


def test_manifest_record_removal_is_detected(tmp_path: Path) -> None:
    report = AttackDatasetGenerator().generate(_pgd_config(tmp_path))
    manifest_path = report.root / "manifest.jsonl"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    manifest_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    inspected = inspect_generated_dataset(report.root)
    assert inspected["count_matches"] is False
    assert inspected["manifest_hash_matches"] is False
    assert inspected["valid"] is False
    with pytest.raises(ValueError, match="manifest count"):
        get_dataset("generated_dataset", root=str(report.root))


def test_different_output_config_gets_new_generation_id(tmp_path: Path) -> None:
    without_preview = _pgd_config(tmp_path)
    with_preview = without_preview.model_copy(update={"preview": True})
    first = AttackDatasetGenerator().generate(without_preview)
    second = AttackDatasetGenerator().generate(with_preview)
    assert first.generation_id != second.generation_id
    assert first.root != second.root


@pytest.mark.parametrize(
    ("attack_name", "attack_params", "needs_surrogate"),
    [
        ("fgsm", {}, True),
        ("pgd", {"steps": 1, "restarts": 1}, True),
        ("mi_fgsm", {"steps": 1}, True),
        (
            "cw_l2",
            {"iterations": 1, "binary_search_steps": 1},
            True,
        ),
        ("tog", {"steps": 1, "variant": "vanishing"}, True),
        ("dag", {"iterations": 1}, True),
        pytest.param(
            "sam2_pgd",
            {"steps": 1},
            True,
            marks=pytest.mark.skip(reason="requires the optional SAM2 segmentation surrogate"),
        ),
        (
            "dpatch",
            {"eot": False, "allow_builtin_patch": True},
            False,
        ),
        (
            "thys_patch",
            {
                "eot": False,
                "target_label": "Pedestrian",
                "allow_builtin_patch": True,
            },
            False,
        ),
    ],
)
def test_each_group_de_attack_exports_reloadable_dataset(
    tmp_path: Path,
    attack_name: str,
    attack_params: dict[str, object],
    needs_surrogate: bool,
) -> None:
    config = AttackGenerationConfig(
        dataset_name="synthetic_shapes",
        dataset_params={"n_samples": 1, "seed": 1},
        attack_name=attack_name,
        attack_params=attack_params,
        severities=[1],
        surrogate=SurrogateConfig(name="blob_detector") if needs_surrogate else None,
        output_dir=str(tmp_path),
        preview=False,
    )
    report = AttackDatasetGenerator().generate(config)
    loaded = get_dataset("generated_dataset", root=str(report.root)).load()
    assert report.n_variants == 1
    assert len(loaded) == 1
    assert loaded[0].image.dtype == np.float32
    assert inspect_generated_dataset(report.root)["valid"] is True
