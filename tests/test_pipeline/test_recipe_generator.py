from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.attacks.recipes import AttackRecipe, AttackRecipeStep
from src.core.hashing import array_digest
from src.datasets import get_dataset
from src.pipeline.cache import GenerationCache, prediction_cache_key
from src.pipeline.generator import (
    AttackDatasetGenerator,
    AttackGenerationConfig,
    RecipeGenerationConfig,
)


def _recipe() -> AttackRecipe:
    names = ("fog", "object_occlusion", "jpeg_compression")
    return AttackRecipe(
        name="composed-fixture",
        steps=tuple(
            AttackRecipeStep(
                position=position,
                attack_name=name,
                implementation_version="1.0.0",
                severity=2,
                seed=100 + position,
                expected_cost=1.0,
            )
            for position, name in enumerate(names)
        ),
    )


def _config(tmp_path: Path) -> RecipeGenerationConfig:
    return RecipeGenerationConfig(
        dataset_name="synthetic_shapes",
        dataset_params={"n_samples": 1, "seed": 44},
        logical_source_id="synthetic-fixture",
        recipe=_recipe(),
        seed=195,
        output_dir=str(tmp_path),
        intended_use="training",
        preview=False,
    )


def test_composed_recipe_generation_round_trips_complete_provenance(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    report = AttackDatasetGenerator().generate(config)
    loaded = get_dataset("generated_dataset", root=str(report.root)).load()
    descriptor = json.loads(
        (report.root / "dataset.json").read_text(encoding="utf-8")
    )
    record = json.loads(
        (report.root / "manifest.jsonl").read_text(encoding="utf-8").strip()
    )

    assert report.n_variants == 1
    assert len(loaded) == 1
    assert descriptor["format"] == "advertest-generated-v3"
    assert descriptor["status"] == "complete"
    assert descriptor["lineage_report_hash"]
    assert record["recipe_hash"] == config.recipe.recipe_hash
    assert [step["attack_name"] for step in record["ordered_steps"]] == [
        "fog",
        "object_occlusion",
        "jpeg_compression",
    ]
    assert all(step["implementation_version"] == "1.0.0" for step in record["ordered_steps"])
    assert all(isinstance(step["derived_seed"], int) for step in record["ordered_steps"])
    assert len(record["intermediate_paths"]) == 3
    for path, expected_hash in zip(
        record["intermediate_paths"],
        record["intermediate_hashes"],
        strict=True,
    ):
        array = np.load(report.root / path, allow_pickle=False)
        assert array_digest(array) == expected_hash
    assert record["source_hash"]
    assert record["ground_truth_hash"]
    assert record["intended_use"] == "training"
    assert record["validation_status"] == "passed"
    assert record["status"] == "complete"
    assert loaded[0].boxes
    assert loaded[0].meta["generation"]["recipe_hash"] == config.recipe.recipe_hash


def test_recipe_generation_resumes_only_hash_valid_intermediates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = AttackDatasetGenerator().generate(config)
    second = AttackDatasetGenerator().generate(config)
    assert second.resumed_variants == 1

    record = json.loads(
        (first.root / "manifest.jsonl").read_text(encoding="utf-8").strip()
    )
    intermediate = first.root / record["intermediate_paths"][0]
    np.save(intermediate, np.zeros((2, 2, 3), dtype=np.float32), allow_pickle=False)

    repaired = AttackDatasetGenerator().generate(config)

    assert repaired.resumed_variants == 0
    restored = np.load(intermediate, allow_pickle=False)
    assert array_digest(restored) == record["intermediate_hashes"][0]


def test_generation_and_prediction_cache_keys_have_separate_provenance() -> None:
    generation = GenerationCache.key(
        dataset_version_id="dataset-1",
        source_hash="source-1",
        recipe_hash="recipe-1",
        implementation_versions=("1.0.0", "2.0.0"),
        seed=195,
        surrogate_version="surrogate-1",
    )
    prediction = prediction_cache_key(
        generated_output_hash="output-1",
        model_version="model-1",
        checkpoint_hash="checkpoint-1",
        preprocessing_version="preprocess-1",
        thresholds={"score": 0.25},
    )

    assert generation != prediction
    assert generation == GenerationCache.key(
        dataset_version_id="dataset-1",
        source_hash="source-1",
        recipe_hash="recipe-1",
        implementation_versions=("1.0.0", "2.0.0"),
        seed=195,
        surrogate_version="surrogate-1",
    )


def test_legacy_config_converts_to_one_step_recipes() -> None:
    config = AttackGenerationConfig(
        dataset_name="synthetic_shapes",
        attack_name="gaussian_noise",
        severities=[1, 3],
    )

    recipes = config.to_recipes(implementation_version="1.0.0")

    assert len(recipes) == 2
    assert all(len(recipe.steps) == 1 for recipe in recipes)
    assert [recipe.steps[0].severity for recipe in recipes] == [1, 3]
