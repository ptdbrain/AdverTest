"""Attack catalog (plan §2).

Auto-discovery means there is no list to edit when someone adds an attack:
drop ``src/attacks/<group>/<your_attack>.py``, decorate the class with
``@ATTACKS.register``, and it shows up in the catalog, the API, the CLI, the
report heatmap, and the contract tests.
"""

from __future__ import annotations

from src.attacks.base import AttackContext, AttackParams, BaseAttack, ModelRequiredError
from src.core.registry import Registry, discover
from src.core.types import AttackGroup, CostClass, Modality

ATTACKS: Registry[BaseAttack] = Registry("attack")

_loaded = False


def load_attacks() -> Registry[BaseAttack]:
    """Import every attack module once, then return the populated registry."""
    global _loaded
    if not _loaded:
        _loaded = True
        discover(__name__)
    return ATTACKS


def get_attack(name: str, **params: object) -> BaseAttack:
    """Instantiate a registered attack with validated parameters."""
    return load_attacks().get(name)(**params)  # type: ignore[arg-type]


def select_attacks(
    *,
    group: AttackGroup | None = None,
    modality: Modality | None = None,
    cost_class: CostClass | None = None,
    supports_gradients: bool | None = None,
) -> list[type[BaseAttack]]:
    """Filter the catalog; ``supports_gradients`` mirrors the model's capability."""
    selected = load_attacks().values()
    if group is not None:
        selected = [attack for attack in selected if attack.group == group]
    if modality is not None:
        selected = [attack for attack in selected if attack.modality == modality]
    if cost_class is not None:
        selected = [attack for attack in selected if attack.cost_class == cost_class]
    if supports_gradients is False:
        selected = [attack for attack in selected if not attack.needs_gradients]
    return selected


__all__ = [
    "ATTACKS",
    "AttackContext",
    "AttackParams",
    "BaseAttack",
    "ModelRequiredError",
    "get_attack",
    "load_attacks",
    "select_attacks",
]
