from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.recipes import RandomNRequest
from src.services.person_d import PersonDServices


def test_person_d_facade_returns_stable_recipe_contracts() -> None:
    load_attacks()
    services = PersonDServices.default()
    recipes = services.recipes.sample(
        RandomNRequest(count=2, steps_per_recipe=1, seed=195)
    )

    assert len(recipes) == 2
    assert recipes[0].recipe_hash
    validation = services.recipes.validate(recipes[0])
    assert validation.valid is True
    assert services.recipes.catalog is ATTACK_CATALOG
