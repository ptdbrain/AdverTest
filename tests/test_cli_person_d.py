import json
from pathlib import Path

from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.recipes import RandomNRequest, RecipeBuilder
from src.cli import main


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_recipe_validate_and_sample_emit_json(tmp_path: Path, capsys) -> None:
    load_attacks()
    recipe = RecipeBuilder().random_n(
        RandomNRequest(count=1, steps_per_recipe=1, seed=195), ATTACK_CATALOG
    )[0]
    validate_path = tmp_path / "validate.json"
    write_json(validate_path, {"recipe": recipe.model_dump(mode="json")})
    assert main(["recipe-validate", "--config", str(validate_path)]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    sample_path = tmp_path / "sample.json"
    write_json(sample_path, RandomNRequest(count=2, steps_per_recipe=1, seed=195).model_dump(mode="json"))
    assert main(["recipe-sample", "--config", str(sample_path)]) == 0
    assert len(json.loads(capsys.readouterr().out)["recipes"]) == 2


def test_person_d_commands_are_registered(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit:
        pass
    output = capsys.readouterr().out
    for command in (
        "dataset-ingest",
        "dataset-split",
        "dataset-validate",
        "recipe-validate",
        "recipe-sample",
        "build-training-dataset",
        "benchmark-run",
        "compare-models",
        "training-estimate",
        "training-run",
    ):
        assert command in output
