"""Registry behaviour — the mechanism that keeps parallel contributions safe."""

from __future__ import annotations

import pytest

from src.adapters import load_adapters
from src.attacks import load_attacks
from src.core.registry import (
    NAME_PATTERN,
    Registry,
    RegistryConflictError,
    UnknownPluginError,
)
from src.datasets import load_datasets


def test_duplicate_name_fails_loudly() -> None:
    registry: Registry[object] = Registry("widget")

    class First:
        name = "shared_name"

    class Second:
        name = "shared_name"

    registry.register(First)
    with pytest.raises(RegistryConflictError, match="already registered"):
        registry.register(Second)


def test_re_registering_the_same_class_is_idempotent() -> None:
    """Module reloads (uvicorn --reload, pytest imports) must not explode."""
    registry: Registry[object] = Registry("widget")

    class Widget:
        name = "some_widget"

    registry.register(Widget)
    registry.register(Widget)
    assert len(registry) == 1


def test_invalid_name_is_rejected() -> None:
    registry: Registry[object] = Registry("widget")

    class BadName:
        name = "Not A Slug"

    with pytest.raises(ValueError, match="snake_case"):
        registry.register(BadName)


def test_unknown_name_lists_alternatives() -> None:
    registry: Registry[object] = Registry("widget")

    class Widget:
        name = "known_widget"

    registry.register(Widget)
    with pytest.raises(UnknownPluginError, match="known_widget"):
        registry.get("missing_widget")


def test_discovery_skips_private_modules() -> None:
    """``_template.py`` must never show up in a catalog."""
    assert "template_attack" not in load_attacks()
    assert "template_model" not in load_adapters()


def test_every_registered_name_is_a_slug() -> None:
    for registry in (load_attacks(), load_adapters(), load_datasets()):
        for name in registry.names():
            assert NAME_PATTERN.match(name), f"{name!r} is not snake_case"
