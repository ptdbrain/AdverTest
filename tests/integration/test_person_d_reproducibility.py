from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.recipes import RandomNRequest, RecipeBuilder


def test_seeded_recipe_generation_is_reproducible() -> None:
    load_attacks()
    request = RandomNRequest(count=3, steps_per_recipe=2, seed=195)
    first = RecipeBuilder().random_n(request, ATTACK_CATALOG)
    second = RecipeBuilder().random_n(request, ATTACK_CATALOG)
    assert [item.recipe_hash for item in first] == [item.recipe_hash for item in second]
