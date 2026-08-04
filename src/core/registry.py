"""Plugin registry with package auto-discovery.

This is the piece that lets several people add attacks in parallel without
merge conflicts: a plugin registers itself from its own module via a decorator,
and :func:`discover` imports every module in the package at start-up. There is
no central list of plugins to edit, so two contributors only ever collide if
they pick the *same* plugin name — which raises :class:`RegistryConflictError`
loudly instead of silently shadowing.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from typing import Generic, TypeVar

T = TypeVar("T")

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,47}$")


class RegistryConflictError(ValueError):
    """Two plugins claimed the same name."""


class UnknownPluginError(KeyError):
    """Requested plugin name is not registered."""


class Registry(Generic[T]):
    """Name -> plugin class mapping for one plugin kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._plugins: dict[str, type[T]] = {}

    def register(self, plugin: type[T]) -> type[T]:
        """Class decorator: ``@ATTACKS.register``."""
        name = getattr(plugin, "name", "")
        if not isinstance(name, str) or not NAME_PATTERN.match(name):
            raise ValueError(
                f"{plugin.__name__}.name must be snake_case (3-48 chars, [a-z0-9_]), got {name!r}"
            )
        existing = self._plugins.get(name)
        if existing is not None and existing is not plugin:
            raise RegistryConflictError(
                f"{self.kind} name {name!r} already registered by "
                f"{existing.__module__}.{existing.__name__}; pick another name"
            )
        self._plugins[name] = plugin
        return plugin

    def get(self, name: str) -> type[T]:
        try:
            return self._plugins[name]
        except KeyError:
            available = ", ".join(self.names()) or "<none>"
            raise UnknownPluginError(f"unknown {self.kind} {name!r}; available: {available}") from None

    def names(self) -> list[str]:
        return sorted(self._plugins)

    def values(self) -> list[type[T]]:
        return [self._plugins[name] for name in self.names()]

    def items(self) -> list[tuple[str, type[T]]]:
        return [(name, self._plugins[name]) for name in self.names()]

    def __contains__(self, name: object) -> bool:
        return name in self._plugins

    def __len__(self) -> int:
        return len(self._plugins)


def _reraise(name: str) -> None:
    """Never let a broken plugin module be skipped silently."""
    raise ImportError(f"failed to import plugin package {name!r}")


def discover(package_name: str) -> None:
    """Import every public submodule of ``package_name`` so decorators run.

    Modules and packages whose name starts with ``_`` are skipped, which is how
    ``_template.py`` stays out of the catalog.
    """
    package = importlib.import_module(package_name)
    paths = list(getattr(package, "__path__", []))
    if not paths:
        return
    for module in pkgutil.walk_packages(paths, prefix=f"{package_name}.", onerror=_reraise):
        parts = module.name.split(".")
        if any(part.startswith("_") for part in parts):
            continue
        importlib.import_module(module.name)
